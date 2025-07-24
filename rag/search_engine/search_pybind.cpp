#include <pybind11/pybind11.h>
#include <string>
#include "searchEngine.h"

namespace py = pybind11;

static SearchEngine engine;

void load() {
    engine.load();
}

std::string search(const std::string& query) {
    return engine.search(query);
}

PYBIND11_MODULE(search_engine, m) {
    m.def("load", &load, "Load the index");
    m.def("search", &search, "Search the query and return result");
}
