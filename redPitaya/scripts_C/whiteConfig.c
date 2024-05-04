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
  uint32_t gain     = strtoul(argv[1],&endptr,10);


  if((fd = open(name, O_RDWR)) < 0) {
    perror("open");
    return 1;
  }
  cfg = mmap(NULL, sysconf(_SC_PAGESIZE), /* map the memory */
             PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0x42000000);

  *((uint32_t *)(cfg + 16)) = gain;   // gainSpring

  return 0;
}