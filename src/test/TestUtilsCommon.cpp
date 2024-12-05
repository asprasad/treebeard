#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <libgen.h>
#include <sstream>
#include <unistd.h>

#include "ExecutionHelpers.h"
#include "TestUtilsCommon.h"

namespace TreeBeard {
namespace test {

std::vector<double> getNextLineAndSplitIntoTokens(std::istream &str,
                                                  char delimiter) {
  std::vector<double> result;
  std::string line;
  std::getline(str, line);

  std::stringstream lineStream(line);
  std::string cell;

  while (std::getline(lineStream, cell, delimiter)) {
    result.push_back(std::stof(cell));
  }
  // This checks for a trailing comma with no data after it.
  if (!lineStream && cell.empty()) {
    // If there was a trailing comma then add an empty element.
    result.push_back(NAN);
  }
  return result;
}

TestCSVReader::TestCSVReader(const std::string &filename, int32_t numRows,
                             bool hasNumRows, char delimiter) {
  std::ifstream fin(filename);
  assert(fin);
  if (hasNumRows) {
    fin >> numRows;
    std::string line;
    std::getline(fin, line);
  }
  int numRowsRead = 0;
  while (!fin.eof() && (numRows == -1 || numRowsRead < numRows)) {
    auto row = getNextLineAndSplitIntoTokens(fin, delimiter);
    m_data.push_back(row);
    ++numRowsRead;
  }
}

std::string GetTreeBeardRepoPath() {
  // char exePath[PATH_MAX];
  // memset(exePath, 0, sizeof(exePath));
  // if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
  //   return std::string("");
  // // std::cout << "Calculated executable path : " << exePath << std::endl;
  // char *execDir = dirname(exePath);
  std::string treebeardBuildDir = TREEBEARD_BUILD_DIR;
  char buildDir[PATH_MAX];
  strcpy(buildDir, treebeardBuildDir.c_str()); // dirname(execDir);
  char *repoPath = dirname(buildDir);
  return repoPath;
}

std::string GetXGBoostModelPath(const std::string &modelFileName) {
  std::filesystem::path tbRepoPath(GetTreeBeardRepoPath());
  tbRepoPath.append("xgb_models").append(modelFileName);
  auto xgboostModelPath = tbRepoPath.string();
  return xgboostModelPath;
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
} // namespace test
} // namespace TreeBeard