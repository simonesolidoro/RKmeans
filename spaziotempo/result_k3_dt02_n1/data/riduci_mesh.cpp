#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>

int main() {
    std::ifstream in("nodes.csv");
    std::ofstream out("nodes_scaled.csv");

    if (!in || !out) {
        std::cerr << "Errore apertura file\n";
        return 1;
    }

    out << std::fixed << std::setprecision(17); // 6 decimali, con arrotondamento

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) { // opzionale: salta righe vuote
            out << "\n";
            continue;
        }

        std::stringstream ss(line);
        std::string field;
        bool first = true;

        while (std::getline(ss, field, ',')) {
            double val = std::stod(field) * 0.5;
            if (!first) out << ",";
            out << val;
            first = false;
        }
        out << "\n";
    }

    return 0;
}

