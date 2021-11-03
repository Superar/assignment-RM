#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NMAX 300

FILE *fp;

int M[NMAX][NMAX];
int p[NMAX];
int found;
double cpu_time_used;
clock_t start, end;
int seed, max_time;

void shuffle(int n, int seed) {
  int i;

  if (n > 1) {
    srand(seed);
    for (i = 0; i <= n - 1; i++) {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = p[j];
      p[j] = p[i];
      p[i] = t;
    }
  }
}

double backtrack(int a[], int pos, int col, int n) {
  int i;
  int c[NMAX + 1];

  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  /* complete solution found*/
  if (pos == n) {
    found = 1;
    return cpu_time_used;
  }
  if (cpu_time_used > max_time) {
    found = -1;
    return cpu_time_used;
  }

  for (i = 1; i <= col; i++)
    c[i] = 1;

  for (i = 0; i < pos; i++) {
    if (M[p[pos]][p[i]] == 1) {
      c[a[p[i]]] = 0;
    }
  }

  for (i = 1; i <= col; i++) {
    if (c[i] == 1) {
      a[p[pos]] = i;
      cpu_time_used = backtrack(a, pos + 1, col, n);
      if ((found == 1) || (found == -1))
        return cpu_time_used;
    }
  }
  return cpu_time_used;
}

int main(int argc, char *argv[]) {
  int i, j, x, y, n, m, c;
  int a[NMAX];

  seed = atoi(argv[1]);
  max_time = atoi(argv[2]);
  fp = fopen(argv[3], "r");

  if (fp == NULL) {
    perror("Error while opening the file.\n");
    exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d %d", &n, &m);
  for (i = 0; i < n; i++) {
    a[i] = 0;
    p[i] = i;
    for (j = 0; j < n; j++)
      M[i][j] = 0;
  }
  shuffle(n, seed);

  for (i = 0; i < m; i++) {
    fscanf(fp, "%d %d", &x, &y);
    M[x - 1][y - 1] = M[y - 1][x - 1] = 1;
  }
  start = clock();

  c = n;
  while (--c) {
    found = 0;
    a[p[0]] = 1;
    cpu_time_used = backtrack(a, 1, c, n);
    if (found == 0) { // found the minimum with c+1
      printf("%d ", c + 1);
      printf("%lf \n", cpu_time_used);
      break;
    } else if (found == -1) { // amount of time exceeded
      printf("-1 ");
      printf("%lf \n", cpu_time_used);
      break;
    }
  }
  if (c == 0) {
    // If c is 0, then 1 was possible.
    printf("1 %lf \n", cpu_time_used);
  }

  return 0;
}
