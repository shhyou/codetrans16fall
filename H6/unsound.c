#include <stdio.h>
#include <CAT.h>

int main(int argc, char **argv) {
  CATData x = CAT_create_signed_value(5), y = CAT_create_signed_value(8), *p;
  if (argc > 5) p = &x; else p = &y;
  *p = CAT_create_signed_value(10);
  printf("%lld %lld\n", CAT_get_signed_value(x), CAT_get_signed_value(y));
}
