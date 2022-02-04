#include "TestUtilsCommon.h"
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <libgen.h>
#include <climits>
#include <cstring>

namespace TreeBeard
{
namespace test
{

std::vector<double> getNextLineAndSplitIntoTokens(std::istream& str) {
    std::vector<double> result;
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    while(std::getline(lineStream, cell, ',')) {
        result.push_back(std::stof(cell));
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.push_back(NAN);
    }
    return result;
}

TestCSVReader::TestCSVReader(const std::string& filename) {
  std::ifstream fin(filename);
  assert(fin);
  while (!fin.eof()) {
    auto row = getNextLineAndSplitIntoTokens(fin);
    m_data.push_back(row);
  }
}

std::string GetTreeBeardRepoPath() {
  char exePath[PATH_MAX];
  memset(exePath, 0, sizeof(exePath)); 
  if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
    return std::string("");
  // std::cout << "Calculated executable path : " << exePath << std::endl;
  char *execDir = dirname(exePath);
  char *buildDir = dirname(execDir);
  char* repoPath = dirname(buildDir);
  return repoPath;
}

std::string GetTempFilePath() {
  char filename[] = "treebeard_temp_XXXXXX";
  auto fd = mkstemp(filename);
  close(fd);
  return std::string(filename);
}

std::string GetGlobalJSONNameForTests() {
  char exePath[PATH_MAX];
  memset(exePath, 0, sizeof(exePath)); 
  if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
    return std::string("");
  char *execDir = dirname(exePath);
  return std::string(execDir) + "/treebeard_test.json";
}

} // test 
} // TreeBeard