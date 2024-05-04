#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int fd;
  void *cfg;
  char *name = "/dev/mem";
  char *endptr;
  /* enables and 1-bit selectors */
  uint32_t enable_biquad_0 = strtoul(argv[1],&endptr,10);
  uint32_t enable_PID      = strtoul(argv[2],&endptr,10);
  uint32_t setPoint_SEL    = strtoul(argv[3],&endptr,10);
  /* biquad_0 gains */
  uint32_t a1_biquad_0     = strtoul(argv[4],&endptr,10);
  uint32_t a2_biquad_0     = strtoul(argv[5],&endptr,10);
  uint32_t b0_biquad_0     = strtoul(argv[6],&endptr,10);
  uint32_t b1_biquad_0     = strtoul(argv[7],&endptr,10);
  uint32_t b2_biquad_0     = strtoul(argv[8],&endptr,10);
  /* PID parameters */
  uint32_t setPoint_RAM    = strtoul(argv[9],&endptr,10);
  uint32_t low_bound       = strtoul(argv[10],&endptr,10);
  uint32_t high_bound      = strtoul(argv[11],&endptr,10);
  uint32_t gain_p          = strtoul(argv[12],&endptr,10);
  uint32_t gain_i          = strtoul(argv[13],&endptr,10);
  uint32_t gain_d          = strtoul(argv[14],&endptr,10);
  
  if((fd = open(name, O_RDWR)) < 0) {
    perror("open");
    return 1;
  }
  cfg = mmap(NULL, sysconf(_SC_PAGESIZE), /* map the memory */
             PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0x40000000);

  *((uint32_t *)(cfg + 0)) = 1*enable_biquad_0 + 2*enable_PID + 4*setPoint_SEL;
  *((uint32_t *)(cfg + 4)) = a1_biquad_0;
  *((uint32_t *)(cfg + 8)) = a2_biquad_0;
  *((uint32_t *)(cfg + 12)) = b0_biquad_0;
  *((uint32_t *)(cfg + 16)) = b1_biquad_0;
  *((uint32_t *)(cfg + 20)) = b2_biquad_0;
  *((uint32_t *)(cfg + 24)) = setPoint_RAM;
  *((uint32_t *)(cfg + 28)) = low_bound;
  *((uint32_t *)(cfg + 32)) = high_bound;
  *((uint32_t *)(cfg + 36)) = gain_p;
  *((uint32_t *)(cfg + 40)) = gain_i;
  *((uint32_t *)(cfg + 44)) = gain_d;

  return 0;
}