/*
  This algorithm is unsound for escaped variables. For example,

  int main(int argc, char **argv) {
    CATData x = CAT_create_signed_value(5), y = CAT_create_signed_value(8), *p;
    if (argc > 5) p = &x; else p = &y;
    *p = CAT_create_signed_value(10);
    printf("%lld %lld\n", CAT_get_signed_value(x), CAT_get_signed_value(y));
  }
*/
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/DependenceAnalysis.h"

#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include <string>
#include <vector>
#include <utility>

using namespace llvm;

namespace {
  std::string to_string(const SmallBitVector& b) {
    std::string res = "b";
    for (size_t i = 0; i != b.size(); ++i)
      res += b.test(i)? "1" : "0";
    return res;
  }

  // probably a bad style to use StringRef
  const unsigned int CAT_create = 0, CAT_get = 1, CAT_add = 2, CAT_sub = 3;
  const StringRef catlang[4] = {
    "CAT_create_signed_value",
    "CAT_get_signed_value",
    "CAT_binary_add",
    "CAT_binary_sub"
  };
  const StringRef catgen[] = { catlang[CAT_create], catlang[CAT_add], catlang[CAT_sub] };

  template<typename F>
  void for_each_inst(Function& function, F action) {
    for (auto& block : function)
      for (auto& inst : block)
        action(inst);
  }

  template<typename F>
  void for_each_gen_inst(Function& function, F action) {
    for_each_inst(function, [&action](Instruction& inst) {
      if (auto* pcallinst = dyn_cast<CallInst>(&inst)) {
        if (Function* callee = pcallinst->getCalledFunction()) {
          StringRef name = callee->getName();
          for (auto* p = std::begin(catgen); p != std::end(catgen); ++p)
            if (*p == name) action(inst, pcallinst, name);
        }
      }
    });
  }

  template<typename F>
  void for_each_phi_inst(Function& function, F action) {
    for_each_inst(function, [&action](Instruction& inst) {
      if (auto* pphi = dyn_cast<PHINode>(&inst))
        action(inst, pphi);
    });
  }

  template<typename F>
  void for_each_mem_inst(Function& function, F action) {
    for_each_inst(function, [&action](Instruction& inst) {
      auto* pload = dyn_cast<LoadInst>(&inst);
      auto* pstore = dyn_cast<StoreInst>(&inst);
      if (pload || pstore)
        action(inst, pload, pstore);
    });
  }

  struct ProgAbs {
    ValueMap<Instruction*, int> inst2idx;
    std::vector<Instruction*> idx2inst;
    std::vector<int> idx2var;
    std::vector<bool> forbids;
    int mapdinst_cnt, definst_cnt;
    DependenceAnalysis& da;
    Function& fn;

    int find(int idx) {
      assert(0 <= idx && idx < idx2var.size() && "find out of bound");
      if (idx != idx2var[idx])
        idx2var[idx] = find(idx2var[idx]);
      return idx2var[idx];
    }

    void link(int a, int b) {
      int sa = find(a), sb = find(b);
      if (sa < sb) idx2var[sb] = sa;
      else         idx2var[sa] = sb;
    }

    int get_idx(Value& val) {
      if (auto* pcallinst = dyn_cast<CallInst>(&val)) {
        if (auto* fn = pcallinst->getCalledFunction()) {
          if (fn->getName() == catlang[CAT_create]) {
            assert(inst2idx.count(pcallinst) && "Unknown CAT_create pcallinst in get_idx");
            return find(inst2idx.lookup(pcallinst));
          } else if (fn->getName() == catlang[CAT_add] || fn->getName() == catlang[CAT_sub]) {
            assert(inst2idx.count(pcallinst) && "Unknown CAT_(add|sub) pcallinst in get_idx");
            return find(inst2idx.lookup(pcallinst));
          } else if (fn->getName() == catlang[CAT_get]) {
            return -1;
          } else {
            return -1;
          }
        } else {
          return -1;
        }
      } else if (auto* pphi = dyn_cast<PHINode>(&val)) {
        assert(inst2idx.count(pphi) && "Unknown PHINode pphi in get_idx");
        return find(inst2idx.lookup(pphi));
      } else if (auto* pstore = dyn_cast<StoreInst>(&val)) {
        return get_idx(*pstore->getValueOperand());
      } else if (auto* pload = dyn_cast<LoadInst>(&val)) {
        assert(inst2idx.count(pload) && "Unknown LoadInst pload in get_idx");
        return find(inst2idx.lookup(pload));
      } else if (auto* parg = dyn_cast<Argument>(&val)) {
        return find(mapdinst_cnt);
      } else {
        return -1;
      }
    }

    ProgAbs(DependenceAnalysis& d, Function& f) : da(d), fn(f) {
      // Map interesting instructions to numbers
      mapdinst_cnt = 0;
      definst_cnt = 0;
      for_each_gen_inst(fn, [this](Instruction& inst, CallInst* pcallinst, StringRef name) {
        inst2idx.insert({&inst, mapdinst_cnt});
        idx2inst.push_back(&inst);
        ++mapdinst_cnt;
        ++definst_cnt;
      });
      for_each_inst(fn, [this](Instruction& inst) {
        if (isa<PHINode>(&inst) || isa<LoadInst>(&inst)) {
          inst2idx.insert({&inst, mapdinst_cnt});
          idx2inst.push_back(&inst);
          ++mapdinst_cnt;
        }
      });

      fn.print(errs());

      // Divide instructions into equivalent classes
      // The extra equivalent class `mapdinst_cnt` is for function
      // arguments and those passed to other functions
      idx2var.resize(mapdinst_cnt+1);
      for (int i = 0; i <= mapdinst_cnt; ++i)
        idx2var[i] = i;

      for_each_gen_inst(fn, [this](Instruction& inst, CallInst* pcallinst, StringRef name) {
        // name is not CAT_create => name is CAT_binary_(add|sub)
        if (name == catlang[CAT_create]) return;

        int inst_idx = inst2idx.lookup(&inst), var_idx = get_idx(*pcallinst->getArgOperand(0));
        if (var_idx == -1) return;
        link(inst_idx, var_idx);
      });

      for_each_phi_inst(fn, [this](Instruction& inst, PHINode* pphi) {
        int idx = inst2idx.lookup(&inst);
        for (size_t i = 0; i != pphi->getNumIncomingValues(); ++i) {
          int var_idx = get_idx(*pphi->getIncomingValue(i));
          if (var_idx == -1) continue;
          link(idx, var_idx);
        }
      });

      // Load and Store instructions are linked according to the result of alias analysis
      // if two memory locations may alias, they should be in the same equivalent class
      // Of course, instructions operating on the *same* pointer should always be in the
      // same equivalent class. DependenceAnalysis does not gives us this information, though.
      for_each_mem_inst(fn, [this](Instruction& inst1, LoadInst* pload1, StoreInst* pstore1) {
        Value* ptr1 = pload1? pload1->getPointerOperand() : pstore1->getPointerOperand();
        for_each_mem_inst(fn, [this,&ptr1,&inst1](Instruction& inst2, LoadInst* pload2, StoreInst* pstore2) {
          Value* ptr2 = pload2? pload2->getPointerOperand() : pstore2->getPointerOperand();
          if (ptr1==ptr2 || da.depends(&inst1, &inst2, true)) {
            int idx1 = get_idx(inst1), idx2 = get_idx(inst2);
            if (idx1 != -1 && idx2 != -1)
              link(idx1, idx2);
          }
        });
      });

      // if a CAT_create'd variable escapes (as a function argument, directly or indirectly),
      // merge its equivalent class into the 'escape equivalent class' (mapdinst_cnt)
      // Currently we just give up on functions except several known functions
      SmallPtrSet<Value*, 64> visited;
      for_each_inst(fn, [&visited,this](Instruction& inst) {
        if (auto* pcallinst = dyn_cast<CallInst>(&inst)) {
          // It is passed as argument? escape
          if (Function* callee = pcallinst->getCalledFunction()) {
            if (callee->getName() == "printf") return;
            if (callee->getName() == "malloc") return;
            if (callee->getName() == "free") return;
            for (auto *p = std::begin(catlang); p != std::end(catlang); ++p)
              if (callee->getName() == *p) return;
          }
          for_each_gen_inst(fn, [&pcallinst,this](Instruction& inst, CallInst*, StringRef name) {
            if (name != catlang[CAT_create]) return;
            if (da.depends(&inst, pcallinst, true)) {
              int s = get_idx(inst);
              if (s != -1) link(s, mapdinst_cnt);
            }
          });
        }
      });

      // mark the escaping equivalent class as 'forbidden'
      forbids.resize(mapdinst_cnt+1);
      forbids[find(mapdinst_cnt)] = true;

      this->print();
    }

    void print() {
      for_each_inst(fn, [this](Instruction& inst) {
        int s = get_idx(inst);
        if (s == -1) errs() << "   X ";
        else         errs() << "   " << s << " ";
        if (inst2idx.count(&inst)) {
          int idx = inst2idx.lookup(&inst);
          errs() << (idx<10?" ":"") << idx;
        } else {
          errs() << "  ";
        }
        inst.print(errs());
        errs() << "\n";
      });
    }
 };

  // data structure storing Gens and Kills
  // Roughly no change for H6
  template<typename BitVect>
  struct ReachInfo {
    std::vector<BitVect> kills;
    std::vector<BitVect> gens;
    BitVect none, every;
    Function& fn;
    ProgAbs& pa;

    ReachInfo(Function& f, ProgAbs& p) : fn(f), pa(p) {
      none = BitVect(pa.definst_cnt);
      every = none;
      every.flip();

      // Create GENs information
      gens.resize(pa.definst_cnt, none);
      for_each_gen_inst(fn, [this](Instruction& inst, CallInst* pcallinst, StringRef name) {
        int idx = pa.inst2idx.lookup(&inst);
        gens[idx].set(idx);
      });

      // Create KILLs U {i} information (aka defs(i))
      kills.resize(pa.definst_cnt, none);
      for_each_gen_inst(fn, [this](Instruction& inst, CallInst* pcallinst, StringRef name) {
        int idx = pa.inst2idx.lookup(&inst);
        int s = pa.get_idx(inst);
        if (s < pa.definst_cnt)
          kills[s].set(idx);
      });

      // Create KILLs information (1): copy kills information
      for_each_gen_inst(fn, [this](Instruction& inst, CallInst* pcallinst, StringRef name) {
        int idx = pa.inst2idx.lookup(&inst);
        int s = pa.get_idx(inst);
        if (s != idx) // if this is not the representative of its equivalent class
          kills[idx] = kills[s];
      });

      // Create KILLs information (2): don't kill self
      for_each_gen_inst(fn, [this](Instruction& inst, CallInst* pcallinst, StringRef name) {
        int idx = pa.inst2idx.lookup(&inst);
        kills[idx].reset(idx);
      });

      // Create KILLs information (3): 1 = not killed, 0 = killed
      for (int i = 0; i != pa.definst_cnt; ++i)
        kills[i].flip();
    }

    const BitVect& getGens(Instruction& inst) const {
      auto it = pa.inst2idx.find(&inst);
      return (it == pa.inst2idx.end() || it->second >= pa.definst_cnt)? none : gens[it->second];
    }

    const BitVect& getKills(Instruction& inst) const {
      auto it = pa.inst2idx.find(&inst);
      return (it == pa.inst2idx.end() || it->second >= pa.definst_cnt)? every : kills[it->second];
    }

    ~ReachInfo() {}
  };

  // Roughly no change for H6
  template<typename BitVect>
  struct ReachDefn {
    Function& fn;
    ProgAbs& pa;
    ReachInfo<BitVect> ri;
    ValueMap<Instruction*, BitVect> ins;
    ValueMap<Instruction*, BitVect> outs;

    ReachDefn(Function& f, ProgAbs& p) : fn(f), pa(p), ri(f, p) {
      // initialize mappings to empty IN and OUT sets
      for_each_inst(f, [this](Instruction& inst) {
        ins[&inst] = ri.none;
        outs[&inst] = ri.none;
      });

      // iterate until fixed point
      std::vector<BasicBlock*> Q(f.size());
      SmallPtrSet<BasicBlock*,32> inQ;
      int enq = 0, deq = 0;

      for (BasicBlock& block : fn) {
        Q[enq++] = &block;
        inQ.insert(&block);
        if (enq == Q.size()) enq = 0;
      }
      while (not inQ.empty()) {
        BasicBlock& block = *Q[deq++];
        inQ.erase(&block);
        if (deq == Q.size()) deq = 0;

        BitVect* in = &ins[&block.front()];
        BitVect block_out = outs[block.getTerminator()];
        for (auto& inst : block) {
          ins[&inst] = *in;

          const BitVect& kills = ri.getKills(inst);
          BitVect* out = &outs.find(&inst)->second;
          *out = *in;
          // non CAT lang gen instructions; no gen and no kill
          if (&kills == &ri.every)
            continue;
          *out &= kills;
          *out |= ri.getGens(inst);
          in = out;
        }
        if (block_out != *in) {
          for (BasicBlock* suc : block.getTerminator()->successors()) {
            ins[&suc->front()] |= *in;
            if (not inQ.count(suc)) {
              Q[enq++] = suc;
              inQ.insert(suc);
              if (enq == Q.size()) enq = 0;
            }
          }
        }
      }
    }

    const BitVect& getIns(Instruction& inst) { return ins[&inst]; }
    const BitVect& getOuts(Instruction& inst) { return outs[&inst]; }
  };

  std::vector<std::pair<Instruction*,ConstantInt*>> constant_propagate(DependenceAnalysis& da, Function& f) {
    std::vector<std::pair<Instruction*,ConstantInt*>> to_modify;
    ProgAbs p{da, f};
    ReachDefn<SmallBitVector> r{f, p};
    for_each_inst(f, [&r,&p,&f,&to_modify](Instruction& inst) {
      int varidx = -1;

      // Unless it is a *USE* of a CAT variable, continue
      // CAT_binary_add and CAT_bianry_sub does not count as *USE* since
      // they accept only CAT variables. There is no way to use a constant.
      if (auto* pcallinst = dyn_cast<CallInst>(&inst)) {
        Function* callee = pcallinst->getCalledFunction();
        if (!callee) return;
        if (callee->getName() != catlang[CAT_get]) return;
        varidx = p.get_idx(*pcallinst->getArgOperand(0));
      } else return;

      inst.print(errs()); errs() << "\n";
      errs() << "    set = " << varidx << "; forbids = " << (varidx<0?"X":(p.forbids[varidx]?"true":"false")) << "\n";
      if (varidx == -1 || p.forbids[varidx]) return;

      const SmallBitVector& ins = r.getIns(inst);
      // Check if all definitions are constant, i.e. d = `CAT_create_signed_value(CONST)`
      bool only_const = true;
      ConstantInt *c = nullptr;
      for_each_gen_inst(f, [&](Instruction& inst, CallInst *pcallinst, StringRef name) {
        // If this definition does not reach this point, OK
        if (not ins.test(p.inst2idx[&inst])) return;

        if (catlang[CAT_create] == name) {
          if (varidx != p.find(p.inst2idx.lookup(&inst))) return;
          if (auto* val = dyn_cast<ConstantInt>(pcallinst->getArgOperand(0))) {
            if (c) only_const = (only_const && (c->getSExtValue() == val->getSExtValue()));
            else   c = val;
          } else {
            only_const = false;
          }
        } else { // name is not CAT_create => name is CAT_binary_(add|sub)
          if (varidx != p.find(p.inst2idx.lookup(&inst))) return;
          only_const = false;
        }
      });

      errs() << "    " << to_string(ins);
      for (int idx = ins.find_first(); idx >= 0; idx = ins.find_next(idx)) {
        Instruction *defn = p.idx2inst[idx];
        errs() << "\n    >>";
        defn->print(errs());
      }
      errs() << "\n    " << (only_const? "const" : "non-const") << " ";
      if (c) errs() << c->getSExtValue();
      errs() << "\n\n";

      if (not only_const || not c) return;

      to_modify.emplace_back(&inst, c);
    });
    return to_modify;
  }

  struct CAT : public FunctionPass {
    static char ID;
    CAT() : FunctionPass(ID) {}

    bool doInitialization (Module &M_) override {
      return false;
    }

    bool runOnFunction (Function &f) override {
      if (f.getName() == "internal_check_data") return false;
      for (auto* p = std::begin(catlang); p != std::end(catlang); ++p)
        if (f.getName() == *p) return false;

      errs() << "===================== " << f.getName() << " =====================\n";
      std::vector<std::pair<Instruction*,ConstantInt*>> to_modify {constant_propagate(getAnalysis<DependenceAnalysis>(), f)};
      for (auto& inst_c : to_modify) {
        inst_c.second->print(errs());
        errs() << "  |=>";
        inst_c.first->print(errs());
        errs() << "\n";
        BasicBlock::iterator it{inst_c.first};
        ReplaceInstWithValue(inst_c.first->getParent()->getInstList(), it, inst_c.second);
      }
      return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequiredTransitive<DependenceAnalysis>();
    }
  };
}

// Next there is code to register your pass to "opt"
char CAT::ID = 0;
static RegisterPass<CAT> X("CAT", "EECS 495 CAT H6");

// Next there is code to register your pass to "clang"
static CAT * _PassMaker = NULL;
static RegisterStandardPasses _RegPass1(PassManagerBuilder::EP_OptimizerLast,
    [](const PassManagerBuilder&, legacy::PassManagerBase& PM) {
    if (!_PassMaker) { PM.add(_PassMaker = new CAT());}}); // ** for -Ox
static RegisterStandardPasses _RegPass2(PassManagerBuilder::EP_EnabledOnOptLevel0,
    [](const PassManagerBuilder&, legacy::PassManagerBase& PM) {
    if (!_PassMaker) { PM.add(_PassMaker = new CAT()); }}); // ** for -O0
