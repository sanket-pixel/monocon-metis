#include "utils/calib.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace monocon {

    ProjMatrix parse_calib_p2(const std::string& calib_path) {
        std::ifstream file(calib_path);
        if (!file.is_open()) {
            throw std::runtime_error("parse_calib_p2: cannot open '" + calib_path + "'");
        }

        std::string line;
        while (std::getline(file, line)) {
            const std::string key = "P_rect_02:";
            if (line.rfind(key, 0) == 0) {
                std::istringstream iss(line.substr(key.size()));
                ProjMatrix P{};
                for (float& v : P) {
                    if (!(iss >> v)) {
                        throw std::runtime_error("parse_calib_p2: malformed P_rect_02 line");
                    }
                }
                return P;
            }
        }

        throw std::runtime_error("parse_calib_p2: 'P_rect_02' not found in '" + calib_path + "'");
    }

} // namespace monocon