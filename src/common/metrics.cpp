#include "metrics.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

static std::string timestamp_now()
{
  const auto now = std::chrono::system_clock::now();
  const auto tt = std::chrono::system_clock::to_time_t(now);
  std::ostringstream oss;
  oss << std::put_time(std::localtime(&tt), "%Y%m%d_%H%M%S");
  return oss.str();
}

std::string default_result_path(const std::string &scenario, const std::string &mode)
{
  fs::create_directories("results/" + scenario + "/" + mode);
  return "results/" + scenario + "/" + mode + "/" + timestamp_now() + ".json";
}

void MetricsWriter::write_json(const std::string &path) const
{
  fs::create_directories(fs::path(path).parent_path());
  FILE *file = std::fopen(path.c_str(), "w");
  if (file == nullptr)
    return;

  std::fprintf(file, "{\n");
  std::fprintf(file, "  \"scenario\": \"%s\",\n", scenario_.c_str());
  std::fprintf(file, "  \"mode\": \"%s\",\n", mode_.c_str());
  std::fprintf(file, "  \"params\": {\n");

  size_t pi = 0;
  for (const auto &[key, value] : params_) {
    std::fprintf(file, "    \"%s\": \"%s\"%s\n", key.c_str(), value.c_str(),
                 ++pi < params_.size() ? "," : "");
  }
  std::fprintf(file, "  },\n");
  std::fprintf(file, "  \"metrics\": {\n");

  size_t mi = 0;
  for (const auto &[key, value] : metrics_) {
    std::fprintf(file, "    \"%s\": %.6f%s\n", key.c_str(), value,
                 ++mi < metrics_.size() ? "," : "");
  }
  std::fprintf(file, "  }\n");
  std::fprintf(file, "}\n");
  std::fclose(file);
}
