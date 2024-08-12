#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>
#include <string>

namespace TreeBeard {
namespace Logging {

struct LoggingOptions {
  bool logGenCodeStats;
  bool logTreeStats;
  bool logGeneralMessages;
  LoggingOptions();
  bool ShouldEnableLogging();
};

extern LoggingOptions loggingOptions;
extern bool loggingEnabled;

inline void Log(const std::string &message) {
  if (loggingEnabled)
    std::cout << message << std::endl;
}

} // namespace Logging
} // namespace TreeBeard

#endif // _LOGGER_H_