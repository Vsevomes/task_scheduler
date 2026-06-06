#pragma once

#include <map>
#include <string>

class MetricsWriter {
public:
  void set_scenario(const std::string &scenario) { scenario_ = scenario; }
  void set_mode(const std::string &mode) { mode_ = mode; }
  void set_metric(const std::string &key, double value) { metrics_[key] = value; }
  void set_param(const std::string &key, const std::string &value) { params_[key] = value; }
  void write_json(const std::string &path) const;

private:
  std::string scenario_;
  std::string mode_;
  std::map<std::string, double> metrics_;
  std::map<std::string, std::string> params_;
};

std::string default_result_path(const std::string &scenario, const std::string &mode);
