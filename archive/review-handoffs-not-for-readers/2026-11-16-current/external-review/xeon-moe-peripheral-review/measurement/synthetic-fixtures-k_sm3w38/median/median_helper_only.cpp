#include <algorithm>
#include <vector>
#include <cstdio>
static double median(std::vector<double> x) { std::sort(x.begin(),x.end());return x[x.size()/2]; }
int main() {
  std::vector<double> a={10,11,12,13,14,20,21,22,23,24};
  std::vector<double> b(10,18);
  std::printf("{\"stock_reported\":%.17g,\"candidate_reported\":%.17g,\"reported_ratio\":%.17g}\n", median(a),median(b),median(a)/median(b));
}
