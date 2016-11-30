# Usage:
# CAT_PASS_LIB=<pass-to-compiled-cat-pass-binary> make all
# Example:
#   CAT_PASS_LIB=~/CAT/lib/CAT.so make all
#   CAT_PASS_LIB=~/CAT/lib/CAT.dylib make all

.PHONY: clean all

clean:
	rm -f unsound.bc unsound.ll unsound_CATd.bc unsound_CATd.ll

all:
	clang-3.8 -I../misc -O0 -c -emit-llvm unsound.c
	opt-3.8 -mem2reg unsound.bc -o unsound.bc
	llvm-dis-3.8 unsound.bc
	opt-3.8 -load $(CAT_PASS_LIB) -basicaa -CAT unsound.bc -o unsound_CATd.bc
	llvm-dis-3.8 unsound_CATd.bc
