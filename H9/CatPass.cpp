/*
   H9 - compile time 0.174s, invocations 189
   constant propagation
   read-after-read elimination
   dead code elimination (trivial version)

   yes dead code
           \r-a-r
      const \_yes_|_no__|
       yes  | 189 | 217 |
       no   | 269 | 297 |

   no dead code
           \r-a-r
      const \_yes_|_no__|
       yes  | 191 | 219 |
       no   | 270 | 298 |
*/

#include "llvm/Support/raw_ostream.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include <queue>
#include <tuple>
#include <utility>

// Utility functions: hash functions for the use of std::unordered_map and std::unordered_set
namespace std {
  template<typename T, typename U>
  struct hash<std::tuple<T*,int,U*>> {
    size_t operator()(const std::tuple<T*,int,U*>& a) const {
      return (std::hash<T*>()(std::get<0>(a)) ^ std::hash<U*>()(std::get<2>(a))) + std::get<1>(a);
    }
  };
  template<typename T, typename U>
  struct hash<std::tuple<llvm::BasicBlock::iterator,T*,U>> {
    size_t operator()(const std::tuple<llvm::BasicBlock::iterator,T*,U>& t) const {
      return std::hash<T*>()(std::get<1>(t)) ^ std::hash<U>()(std::get<2>(t));
    }
  };
};

using namespace llvm;

// probably a bad style to use StringRef
const static unsigned int CAT_create = 0, CAT_get = 1, CAT_add = 2, CAT_sub = 3;
const static StringRef catlang[4] = {
  "CAT_create_signed_value",
  "CAT_get_signed_value",
  "CAT_binary_add",
  "CAT_binary_sub"
};

/*
   This program analysis approximates the runtime behavior of a program. The behavior
   is represented by a store mapping all variables to the approximation of its runtime
   value. For this assignment, the approximation of runtime values includes constants
   that appeared in the program and possible destinations of pointers.

   The entire analysis is represented as a transition relation between program states,
   which contains three components:

   - store : addr_t → pvalue_t

     a store mapping addresses to approximation of runtime values

   - stack : addr_t → frame_t

     a stack that approximates whole-program control flow graph which is built on-the-fly.
     frame_t includes a code pointer, call-site (for context sensitivity) and link to previous
     frames. Since we are only using context sensitivity to distinguish the return paths from
     different calls, it might happen that results of different applications of the same
     function be merged superfluously. Though this has already been solve by using a different
     address space for stacks which perfectly approximates function applications, it is still
     too much for this assignment.

     * frame_t : < code_pointer, code_pointer (callsite), P(addr_t) >

   - incoming : addr_t → P(basic blocks)

     For each basic block B, incoming(B) contains the approximation of predecessors of B. This is
     only used to implement the semantics of φ-nodes.

   Pointer arithmetic and indirect function calls are not supported in current implementation
   But it is possible to support indirect function calls by adding function names to pvalue_t
   and approximate the set of functions that a function pointer may points to at the same time.
*/
namespace {
  /*
     An address is composed of current location, 1 call-site and a tag
     tag 0 is for the value of this instruction while tag 1, 2, ... are
     addresses allocated/used by this instruction.

     Current address space:
     - <Function*, callsite>: stack frames
     - <BasicBlock*, callsite>: 'incomings' for φ-nodes
     - <Instruction*, callsite> and <Argument*, callsite>: store values
  */
  using addr_t = std::tuple<Value*, int, Instruction*>;

  /* Where P(_) means power set, constants denote those that appear in the program and
     top = might be anything.

                top
           /           \
   P({constants})   P({addrs})
           \           /
                { }
  */
  struct pvalue_t {
    bool is_top;
    std::unordered_set<ConstantInt*> consts;
    std::unordered_set<addr_t> addrs;
    bool is_bottom() const { return !this->is_top && this->consts.empty() && this->addrs.empty(); }
    pvalue_t(bool is_top_ = false) : is_top(is_top_) {}

    bool operator!=(const pvalue_t& that) const { return not(*this == that); }
    bool operator==(const pvalue_t& that) const {
      return this->is_top == that.is_top && this->consts == that.consts && this->addrs == that.addrs;
    }

    pvalue_t& operator+=(const pvalue_t& that) {
      this->is_top = this->is_top || that.is_top;
      if (this->is_top) {
        this->consts.clear();
        this->addrs.clear();
      } else {
        for (const auto& c : that.consts) this->consts.insert(c);
        for (const auto& a : that.addrs)  this->addrs.insert(a);
      }
      return *this;
    }
  };

  //                             return point       call site     link
  using frame_t = std::tuple<BasicBlock::iterator, Instruction*, addr_t>;

  BasicBlock::iterator get_code(const frame_t& frame) { return std::get<0>(frame); }
  Instruction* get_callsite(const frame_t& frame) { return std::get<1>(frame); }
  addr_t get_link(const frame_t& frame) { return std::get<2>(frame); }

  // A strange queue that avoid inserting repeated basic blocks as well as staging
  // between different rounds of iterations. For example, if the basic blocks being
  // 'enqueue'ing has already been alive_creates in this round, defer its 'dequeue'ing
  // until next round. This is for termination checking.
  struct strangequeue_t {
    using T = std::tuple<BasicBlock::iterator, Instruction*, addr_t>;
    using container_t = std::unordered_set<T>;
    container_t curr_pending, alive_creates, next_pending;
    std::deque<container_t::iterator> curr_order, next_order;

    bool empty() const { return curr_order.empty(); }
    const T& top() { alive_creates.insert(*curr_order.front()); return *curr_order.front(); }

    void flip() {
      assert(curr_pending.empty() && curr_order.empty() && "Flip while queue is not empty");
      container_t().swap(alive_creates);
      curr_pending.swap(next_pending);
      curr_order.swap(next_order);
    }

    // Some heuristics for the ordering of basic blocks. For function calls, push continuations
    // to the front of the queue. For branches, push continuations to the rear of the queue.
    void push(const T& t, bool isFunctionCall) {
      bool not_alive_creates = this->alive_creates.count(t) == 0;
      container_t& pending = not_alive_creates? this->curr_pending : this->next_pending;
      std::deque<container_t::iterator>& order = not_alive_creates? this->curr_order : this->next_order;

      std::pair<container_t::iterator, bool> res = pending.insert(t);
      if (res.second) { // newly inserted (not in the queue)
        if (isFunctionCall) order.push_front(res.first);
        else order.push_back(res.first);
      }
    }

    void pop() {
      curr_pending.erase(curr_order.front());
      curr_order.pop_front();
    }
  };

  struct analysis {
    Module& m;
    bool has_unsupported_instruction;

    // Assuming that Value* are all distinct accross the module
    std::unordered_map<addr_t, pvalue_t> store;
    std::unordered_map<addr_t, std::unordered_set<BasicBlock*>> incomings;
    std::unordered_map<addr_t, std::unordered_set<frame_t>> stack;

    pvalue_t store_at_or_top(const addr_t& l) {
      if (auto* c = dyn_cast<ConstantInt>(std::get<0>(l))) {
        pvalue_t x; x.consts.insert(c);
        return x;
      } else if (auto* c = dyn_cast<ConstantPointerNull>(std::get<0>(l))) {
        return pvalue_t(); // points to NULL (no address at all)
      } else {
        return store.count(l) == 0? pvalue_t(true) : store.at(l);
      }
    }

    bool store_update(const addr_t& l, pvalue_t x, bool must_exist = true) {
      if (not must_exist && store.count(l) == 0) {
        store.insert({l, std::move(x)});
        return true;
      } else {
        pvalue_t before = store.at(l);
        store.at(l) += x;
        return before != store.at(l);
      }
    }

    analysis(Module& m_) : m(m_) {
      has_unsupported_instruction = false;
      for (Function& f : m) {
        for (BasicBlock& b : f) {
          for (Instruction& i : b) {
            Instruction* inst = &i;
            const char* message = nullptr;
            if (isa<SelectInst>(inst)) message = "use PHINode; SelectInst";
            if (isa<SwitchInst>(inst)) message = "use BranchInst; SwitchInst";
            if (isa<ResumeInst>(inst)) message = "exception ResumeInst";
            if (isa<CatchReturnInst>(inst)) message = "exception CatchReturnInst";
            if (isa<CatchSwitchInst>(inst)) message = "exception CatchSwitchInst";
            if (auto* pcall = dyn_cast<CallInst>(inst))
              if (pcall->getCalledFunction() == nullptr)
                message = "Indirect calls";
            if (message) {
              errs() << message << " not supported\n";
              has_unsupported_instruction = true;
            }
          }
        }
      }
    }

    void init_store_for_function(Function& fn, Instruction* callsite) {
      for (auto it = inst_begin(fn); it != inst_end(fn); ++it) {
        Instruction* inst = &*it;
        addr_t a{inst, 0, callsite};
        if (store.count(a) == 0) {
          // initialize those we care from {} (we're computing least fixed point)
          // ReturnInst and StoreInt has no value
          if (isa<PHINode>(inst) || isa<ICmpInst>(inst) || isa<AllocaInst>(inst)
                || isa<LoadInst>(inst) || isa<BitCastInst>(inst))
          {
            store.insert({a, {}});
          } else if (auto* pcall = dyn_cast<CallInst>(inst)) {
            Function* callee = pcall->getCalledFunction();
            StringRef name = callee->getName();
            if (name == catlang[CAT_create] || name == catlang[CAT_get] || name == "malloc"
                || (not callee->empty() && name != catlang[CAT_add] && name != catlang[CAT_sub]))
            {
              store.insert({a, {}});
            }
          }
        }
      }
    }

    bool compute(Function& fn) {
      if (has_unsupported_instruction) return false;
      strangequeue_t pending;
      {
        addr_t fn_addr{&fn, 0, nullptr};
        stack.insert({fn_addr, {}});
        //                             code start      callsite  stack_top
        pending.push(std::make_tuple(fn.front().begin(), nullptr, fn_addr), true);
        for (Argument& arg : fn.getArgumentList()) {
          addr_t l{&arg, 0, nullptr};
          assert(store.count(l) == 0 && "Why rerunning funtion");
          store.insert({l, pvalue_t(true)});
        }

        init_store_for_function(fn, nullptr);
      }

      // Possible optimization: find a good ordering of continuations. A natural order would be
      // the execution of function call and branches, possibly following the SCC of CFG and
      // call graph. But I am not sure how to check for termination in that case.
      bool changed = true, success = true;
      while (success && changed && not pending.empty()) {
        changed = false;
        while (not pending.empty()) {
          BasicBlock::iterator it;
          Instruction* callsite{nullptr};
          addr_t stack_top;
          std::tie(it, callsite, stack_top) = pending.top(); pending.pop();

          while (success) {
            Instruction* inst{static_cast<Instruction*>(it)};

            if (auto* pcall = dyn_cast<CallInst>(inst)) {
              // No need to check whether callee == nullptr. has_unsupported_instruction should
              // have been true and compute() will return false in that case.
              Function* callee = pcall->getCalledFunction();
              StringRef name = callee->getName();
              if (name == catlang[CAT_create]) {
                // %call = CAT_create_signed_value(x)
                //   a
                // a ↦ {... l ...}, l ↦ {... c ...}, l fresh
                const pvalue_t c_hat = store_at_or_top(addr_t{pcall->getArgOperand(0), 0, callsite});
                addr_t l{pcall, 1, callsite};
                changed |= store_update(l, std::move(c_hat), false);

                pvalue_t x; x.addrs.insert(l);
                addr_t a{pcall, 0, callsite};
                const pvalue_t& y = store.at(a);
                assert((y.is_bottom() || y==x) && "CAT_create_signed_value memory address changed");
                changed |= store_update(a, x);
                ++it;
              } else if (name == catlang[CAT_add] || name == catlang[CAT_sub]) {
                addr_t l{pcall->getArgOperand(0), 0, callsite};
                const pvalue_t& cat = store.at(l);
                assert(cat.consts.empty() && "CAT_(add|sub) type error");
                const pvalue_t& top(true);
                for (const addr_t& a : cat.addrs)
                  changed |= store_update(a, top);
                ++it;
              } else if (name == catlang[CAT_get]) {
                // handle CAT_get_signed_value(NULL)
                const pvalue_t& ls = store_at_or_top(addr_t{pcall->getArgOperand(0), 0, callsite});
                assert(ls.consts.empty() && "CAT_get_signed_value type error");

                addr_t a{pcall, 0, callsite};
                if (ls.is_top) {
                  changed |= store_update(a, pvalue_t(true));
                } else {
                  pvalue_t x;
                  for (const addr_t& l : ls.addrs)
                    x += store_at_or_top(l);
                  changed |= store_update(a, x);
                }
                ++it;
              } else if (callee->empty()) { // TODO: external linkage, malloc
                addr_t a{pcall, 0, callsite};
                if (callee->getName() == "malloc") {
                  addr_t l{pcall, 1, callsite};
                  changed |= store_update(l, pvalue_t(), false);

                  pvalue_t x; x.addrs.insert(l);
                  changed |= store_update(a, x);
                }
                ++it;
              } else {
                // address for the stack frame of callee
                addr_t a{callee, 0, pcall};
                if (stack.count(a) == 0)
                  stack.insert({a, {}});
                stack.at(a).insert(frame_t{it, callsite, stack_top});

                for (Argument& arg : callee->getArgumentList()) {
                  size_t i = arg.getArgNo();

                  const pvalue_t& x = store_at_or_top(addr_t{pcall->getArgOperand(i), 0, callsite});
                  addr_t l{&arg, 0, pcall};
                  changed |= store_update(l, x, false);
                }

                init_store_for_function(*callee, pcall);
                pending.push(std::make_tuple(callee->front().begin(), pcall, a), true);
                break;
              }
            } else if (auto* pret = dyn_cast<ReturnInst>(inst)) {
              std::unordered_set<frame_t>& continuations = stack.at(stack_top);
              if (Value* retval = pret->getReturnValue()) {
                const pvalue_t& x = store_at_or_top(addr_t{retval, 0, callsite});
                for (const frame_t& cont : continuations) {
                  BasicBlock::iterator it = get_code(cont);
                  CallInst* pcall = dyn_cast<CallInst>(static_cast<Instruction*>(it));
                  assert(pcall && "Return to non call instruction");
                  Instruction* callsite = get_callsite(cont);
                  addr_t l{pcall, 0, callsite};
                  changed |= store_update(l, x);
                  ++it;
                  pending.push(std::make_tuple(it, callsite, get_link(cont)), true);
                }
              } else {
                for (const frame_t& cont : continuations)
                  pending.push(std::make_tuple(++get_code(cont), get_callsite(cont), get_link(cont)), true);
              }
              break;
            } else if (auto* pbr = dyn_cast<BranchInst>(inst)) {
              int cnt = 0, succ[2] = {99999999, 99999999};
              if (pbr->isUnconditional()) {
                succ[cnt++] = 0;
              } else {
                Value* cond = pbr->getCondition();
                if (auto* c = dyn_cast<Constant>(cond)) {
                  if (c->isOneValue()) succ[cnt++] = 0;
                  if (c->isZeroValue()) succ[cnt++] = 1;
                } else {
                  const pvalue_t& x = store_at_or_top(addr_t{cond, 0, callsite});
                  // It might be anything or we have no idea whether it is a NULL pointer
                  if (x.is_top || not x.addrs.empty()) {
                    succ[cnt++] = 0;
                    succ[cnt++] = 1;
                  } else { // must be constants
                    for (auto it = std::begin(x.consts); it != std::end(x.consts); ++it) {
                      if ((*it)->isOneValue()) {
                        succ[cnt++] = 0;
                        break;
                      }
                    }
                    for (auto it = std::begin(x.consts); it != std::end(x.consts); ++it) {
                      if ((*it)->isZeroValue()) {
                        succ[cnt++] = 1;
                        break;
                      }
                    }
                  }
                }
              }
              for (int i = 0; i != cnt; ++i) {
                BasicBlock* block = pbr->getSuccessor(succ[i]);
                addr_t b{block, 0, callsite};
                if (incomings.count(b) == 0) incomings.insert({b, {}});
                incomings.at(b).insert(pbr->getParent());
                pending.push(std::make_tuple(block->begin(), callsite, stack_top), false);
              }
              break;
            } else if (auto* pphi = dyn_cast<PHINode>(inst)) {
              const auto& froms = incomings.at(addr_t{pphi->getParent(), 0, callsite});
              pvalue_t x;
              for (size_t i = 0; i != pphi->getNumIncomingValues(); ++i) {
                Value* val = pphi->getIncomingValue(i);
                if (isa<UndefValue>(val)) continue;
                if (froms.count(pphi->getIncomingBlock(i)))
                  x += store_at_or_top(addr_t{val, 0, callsite});
              }
              addr_t a{pphi, 0, callsite}, b{pphi->getParent(), 0, callsite};
              changed |= store_update(a, x);
              ++it;
            } else if (auto* picmp = dyn_cast<ICmpInst>(inst)) {
              CmpInst::Predicate pred = picmp->getPredicate();
              const pvalue_t& v1 = store_at_or_top(addr_t{picmp->getOperand(0), 0, callsite});
              const pvalue_t& v2 = store_at_or_top(addr_t{picmp->getOperand(1), 0, callsite});

              bool result[2] = {false, false};
              if (v1.is_top || v2.is_top || not v1.addrs.empty() || not v2.addrs.empty()) {
                result[false] = true;
                result[true] = true;
              } else {
                for (const ConstantInt* c1 : v1.consts) {
                  auto n1 = c1->getSExtValue(); // int64_t on my computer
                  for (const ConstantInt* c2 : v2.consts) {
                    auto n2 = c2->getSExtValue(); // int64_t on my computer
                    if (pred == CmpInst::ICMP_EQ) {
                      result[n1 == n2] = true;
                    } else if (pred == CmpInst::ICMP_NE) {
                      result[n1 != n2] = true;
                    } else if (pred == CmpInst::ICMP_SGT) {
                      result[n1 > n2] = true;
                    } else if (pred == CmpInst::ICMP_SGE) {
                      result[n1 >= n2] = true;
                    } else if (pred == CmpInst::ICMP_SLT) {
                      result[n1 < n2] = true;
                    } else if (pred == CmpInst::ICMP_SLE) {
                      result[n1 <= n2] = true;
                    } else {
                      result[false] = true;
                      result[true] = true;
                    }
                  }
                }
              }
              pvalue_t x;
              if (result[false]) x.consts.insert(ConstantInt::getFalse(m.getContext()));
              if (result[true]) x.consts.insert(ConstantInt::getTrue(m.getContext()));

              addr_t a{picmp, 0, callsite};
              changed |= store_update(a, x);
              ++it;
            } else if (auto* palloc = dyn_cast<AllocaInst>(inst)) {
              pvalue_t x;
              addr_t l{palloc, 1, callsite};
              x.addrs.insert(l);
              changed |= store_update(l, pvalue_t(), false);

              addr_t a{palloc, 0, callsite};
              changed |= store_update(a, std::move(x));
              ++it;
            } else if (auto* pbitcast = dyn_cast<BitCastInst>(inst)) {
              pvalue_t x = store_at_or_top(addr_t{pbitcast->getOperand(0), 0, callsite});

              addr_t a{pbitcast, 0, callsite};
              changed |= store_update(a, std::move(x));
              ++it;
            } else if (auto* pload = dyn_cast<LoadInst>(inst)) {
              addr_t l{pload->getPointerOperand(), 0, callsite};
              addr_t a{pload, 0, callsite};
              if (store.count(l) == 1 && not store.at(l).is_top) {
                const pvalue_t& p = store.at(l);
                assert(p.consts.empty() && "pload type error");
                pvalue_t x;
                for (const addr_t& b : p.addrs)
                  x += store_at_or_top(b);
                changed |= store_update(a, x);
              } else {
                changed |= store_update(a, pvalue_t(true));
              }

              ++it;
            } else if (auto* pstore = dyn_cast<StoreInst>(inst)) {
              // Possible improvement of LoadInst and StoreInst: use alias analysis
              pvalue_t p = store_at_or_top(addr_t{pstore->getPointerOperand(), 0, callsite});
              if (p.is_top) {
                errs() << "    "; pstore->getPointerOperand()->printAsOperand(errs(), false);
                errs() << " points to unknown addresses";
                success = false;
                break;
              }
              assert(p.consts.empty() && "pstore pointer type error");
              pvalue_t x = store_at_or_top(addr_t{pstore->getValueOperand(), 0, callsite});
              for (const addr_t& b : p.addrs)
                changed |= store_update(b, x);
              ++it;
            } else {
              ++it;
            }
          }
        }
        pending.flip();
      }
      return success;
    }
  };

  void constant_propagate(const std::unordered_map<Value*, pvalue_t>& collapsed_store, Module& m) {
    std::vector<std::pair<Instruction*, ConstantInt*>> to_modify;
    for (const auto& pr : collapsed_store) {
      auto* pcall = dyn_cast<CallInst>(pr.first);
      if (not pcall || pcall->getCalledFunction()->getName() != catlang[CAT_get]) continue;
      assert(pr.second.addrs.empty() && "CAT_get_signed_value type error");
      if (not pr.second.is_top && not pr.second.consts.empty()) {
        bool is_const = true;
        ConstantInt* c = *pr.second.consts.begin();
        for (const auto& c2 : pr.second.consts) {
          if (c->getSExtValue() != c2->getSExtValue()) {
            is_const = false;
            break;
          }
        }
        if (is_const) to_modify.push_back({pcall, c});
      }
    }
    for (auto& inst_c : to_modify) {
      BasicBlock::iterator it{inst_c.first};
      ReplaceInstWithValue(inst_c.first->getParent()->getInstList(), it, inst_c.second);
    }
  }

  template<typename T>
  void set_erase_when_count_is(std::unordered_set<T>& lhs, const std::unordered_set<T>& rhs, size_t c) {
    auto it = lhs.begin();
    while (it != lhs.end()) {
      auto jt = it; ++it;
      if (rhs.count(*jt) == c) lhs.erase(jt);
    }
  }

  void read_read_elim(const std::unordered_map<Value*, pvalue_t>& collapsed_store, Module& m) {
    std::vector<std::pair<CallInst*, CallInst*>> to_modify;
    for (Function& f : m) {
      if (f.empty()) continue;

      std::vector<CallInst*> cat_gets;
      for (auto it = inst_begin(f); it != inst_end(f); ++it)
        if (auto* pcall = dyn_cast<CallInst>(&*it))
          if (pcall->getCalledFunction()->getName() == catlang[CAT_get])
            cat_gets.push_back(pcall);

      std::unordered_map<CallInst*, std::unordered_set<CallInst*>> kills;
      for (auto it = inst_begin(f); it != inst_end(f); ++it) {
        auto* pcall = dyn_cast<CallInst>(&*it);
        if (not pcall) continue;
        StringRef name = pcall->getCalledFunction()->getName();
        if (name == catlang[CAT_create] || name == catlang[CAT_get]) { // do nothing
        } else if (name == catlang[CAT_add] || name == catlang[CAT_sub] || not pcall->getCalledFunction()->empty()) {
          kills.insert({pcall, {}});
          std::unordered_set<CallInst*>& kill = kills.at(pcall);
          std::unordered_set<Value*> cat_creates;
          int numArgs = (name == catlang[CAT_add] || name == catlang[CAT_sub])? 1 : pcall->getNumArgOperands();
          bool is_unknown = false;
          for (int i = 0; i < numArgs && not is_unknown; ++i) {
            Value* arg = pcall->getArgOperand(i);
            if (collapsed_store.count(arg) == 0) continue;
            const pvalue_t& x = collapsed_store.at(arg);
            // improvement: can just
            is_unknown = (is_unknown || x.is_top);
            for (const addr_t& l2 : x.addrs) cat_creates.insert(std::get<0>(l2));
          }
          for (CallInst* pget : cat_gets) {
            bool intersected = is_unknown;
            Value* arg = pget->getArgOperand(0);
            if (collapsed_store.count(arg) == 0) continue;
            const pvalue_t& x = collapsed_store.at(arg);
            intersected = (intersected || x.is_top);
            for (const addr_t& l2 : x.addrs)
              intersected = (intersected || cat_creates.count(std::get<0>(l2)));
            if (intersected) kill.insert(pget);
          }
        }
      }

      std::unordered_map<CallInst*, std::unordered_set<CallInst*>> alives;
      std::unordered_map<BasicBlock*, std::unordered_set<CallInst*>> outs;
      std::unordered_set<BasicBlock*> inQ;
      std::queue<BasicBlock*> Q;
      for (BasicBlock& b : f) {
        inQ.insert(&b);
        Q.push(&b);
        outs.insert({&b, {}});
      }
      while (not Q.empty()) {
        BasicBlock* b = Q.front(); Q.pop(); inQ.erase(b);

        // IN[b] = intersection OUT[preds]
        pred_iterator pi = pred_begin(b);
        std::unordered_set<CallInst*> out;
        if (pi != pred_end(b)) out = outs.at(*pi);
        for (; pi != pred_end(b); ++pi) set_erase_when_count_is(out, outs.at(*pi), 0);
        for (Instruction& i : *b) {
          CallInst* pcall = dyn_cast<CallInst>(&i);
          if (not pcall) continue;
          StringRef name = pcall->getCalledFunction()->getName();
          if (name == catlang[CAT_create]) { // do nothing
          } else if (name == catlang[CAT_get]) {
            alives[pcall] = out;
            auto it = out.begin();
            for (; it != out.end(); ++it)
              if (pcall->getArgOperand(0) == (*it)->getArgOperand(0))
                break;
            if (it == out.end()) out.insert(pcall);
          } else if (name == catlang[CAT_add] || name == catlang[CAT_sub] || not pcall->getCalledFunction()->empty()) {
            set_erase_when_count_is(out, kills.at(pcall), 1);
          }
        }
        if (out != outs.at(b)) {
          outs.at(b).swap(out);
          for (succ_iterator si = succ_begin(b); si != succ_end(b); ++si) {
            if (inQ.count(*si) == 0) {
              inQ.insert(*si);
              Q.push(*si);
            }
          }
        }
      }

      for (CallInst* pcall : cat_gets)
        for (CallInst* pget : alives.at(pcall))
          if (pcall->getArgOperand(0) == pget->getArgOperand(0))
            to_modify.push_back(std::make_pair(pcall, pget));
    }
    for (auto& inst_g : to_modify) {
      BasicBlock::iterator it{inst_g.first};
      ReplaceInstWithValue(inst_g.first->getParent()->getInstList(), it, inst_g.second);
    }
  }

  void dead_code_elim(const std::unordered_map<Value*, pvalue_t>& collapsed_store, Module& m) {
    std::vector<Instruction*> to_modify;
    for (Function& f : m) {
      if (f.empty()) continue;

      std::unordered_set<Value*> alives;
      for (auto it = inst_begin(f); it != inst_end(f); ++it) {
        auto* pcall = dyn_cast<CallInst>(&*it);
        if (not pcall) continue;
        for (size_t i = 0; i != pcall->getNumArgOperands(); ++i) {
          Value* arg = pcall->getArgOperand(i);
          if (not isa<PointerType>(arg->getType())) continue;
          if (collapsed_store.count(arg) == 0) continue;
          const pvalue_t x = collapsed_store.at(arg);
          if (x.is_top) return;
          for (const addr_t& l : x.addrs)
            alives.insert(std::get<0>(l));
        }
      }
      for (auto it = inst_begin(f); it != inst_end(f); ++it)
        if (auto* pcall = dyn_cast<CallInst>(&*it))
          if (pcall->getCalledFunction()->getName() == catlang[CAT_create])
            if (alives.count(pcall) == 0)
              to_modify.push_back(pcall);
    }
    if (not m.getFunction(catlang[CAT_create])) return;
    PointerType* pptype = dyn_cast<PointerType>(m.getFunction(catlang[CAT_create])->getReturnType());
    if (not pptype) return;
    ConstantPointerNull* pnullptr = ConstantPointerNull::get(pptype);
    for (Instruction* inst : to_modify) {
      BasicBlock::iterator it{inst};
      ReplaceInstWithValue(inst->getParent()->getInstList(), it, pnullptr);
    }
  }

  struct CAT : public ModulePass {
    static char ID;

    CAT() : ModulePass(ID) {}

    bool doInitialization (Module&) override { return false; }

    bool runOnModule (Module& m) override {
      if (m.getFunction("main") == nullptr) {
        errs() << "Function 'main' not found\n";
        return false;
      }
      analysis AA(m);
      bool success = AA.compute(*m.getFunction("main"));
      if (not success) { errs() << "Analysis not success.\n"; return false; }
      std::unordered_map<Value*, pvalue_t> collapsed_store;
      for (const auto pr : AA.store) {
        if (std::get<1>(pr.first) != 0) continue;
        Value* v = std::get<0>(pr.first);
        if (collapsed_store.count(v) == 0) collapsed_store.insert({v, pr.second});
        else collapsed_store.at(v) += pr.second;
      }
      constant_propagate(collapsed_store, m);
      read_read_elim(collapsed_store, m);
      dead_code_elim(collapsed_store, m);
      return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
    }
  };

  // All sorts of printing; for debugging
  void printSourceName(Value* val) {
    if (auto* inst = dyn_cast<Instruction>(val)) {
      errs() << inst->getParent()->getParent()->getName() << ".";
    } else if (auto* arg = dyn_cast<Argument>(val)) {
      errs() << arg->getParent()->getName() << ".";
    }
  }

  void printOperandOrFull(Value* v) {
    if (v == nullptr) errs() << "*";
    else if (v->getType()->isVoidTy()) errs() << "??";
    else v->printAsOperand(errs(), false);
  }

  void print(const addr_t& a) {
    // Some addresses might be printed as <badref> because those instructions do not compute values.
    Value* val = std::get<0>(a);
    int tag = std::get<1>(a);
    errs() << "#a{"; printSourceName(val); printOperandOrFull(val);
    if (tag != 0) errs() << "-" << tag;
    errs() << ","; printOperandOrFull(std::get<2>(a)); errs() << "}";
  }

  void print(const pvalue_t& v) {
    if (v.is_top) {
      errs() << "⊤";
    } else {
      errs() << "{ ";
      for (ConstantInt* c : v.consts) {
        c->printAsOperand(errs(), false);
        errs() << " ";
      }
      for (const addr_t& a : v.addrs) {
        print(a);
        errs() << " ";
      }
      errs() << "}";
    }
  }

  void print(BasicBlock::iterator it, Instruction* callsite, addr_t stack_top) {
    print(stack_top); errs() << " "; printOperandOrFull(callsite);
    errs() << " >>> "; it->print(errs()); errs() << "\n";
  }

  void print(const std::unordered_map<addr_t, pvalue_t>& store) {
    std::vector<addr_t> addrs;
    for (auto& it : store)
      addrs.push_back(it.first);
    std::sort(std::begin(addrs), std::end(addrs), [](const addr_t& a1, const addr_t& a2) {
      Instruction *callsite1, *callsite2;
      Value *v1, *v2;
      int tag1, tag2;
      std::tie(v1, tag1, callsite1) = a1;
      std::tie(v2, tag2, callsite2) = a2;
      if (callsite1 != callsite2)
        return std::less<Instruction*>()(callsite1, callsite2);
      else if (v1 != v2)
        return std::less<Value*>()(v1, v2);
      else
        return tag1 < tag2;
    });
    errs() << "\nStore\n";
    for (addr_t& a : addrs) {
      errs() << "σ("; print(a); errs() << ") = "; print(store.at(a)); errs() << "\n";
    }
  }
}

// Next there is code to register your pass to "opt"
char CAT::ID = 0;
static RegisterPass<CAT> X("CAT", "EECS 495 CAT H9");

// Next there is code to register your pass to "clang"
static CAT * _PassMaker = NULL;
static RegisterStandardPasses _RegPass1(PassManagerBuilder::EP_OptimizerLast,
    [](const PassManagerBuilder&, legacy::PassManagerBase& PM) {
    if (!_PassMaker) { PM.add(_PassMaker = new CAT());}}); // ** for -Ox
static RegisterStandardPasses _RegPass2(PassManagerBuilder::EP_EnabledOnOptLevel0,
    [](const PassManagerBuilder&, legacy::PassManagerBase& PM) {
    if (!_PassMaker) { PM.add(_PassMaker = new CAT()); }}); // ** for -O0
