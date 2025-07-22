#include <string>
#include "httplib.h"
#include "searchEngine.h"

int main() {
    httplib::Server server;

    SearchEngine searchEngine;
    searchEngine.load();

    server.Get("/search", [&searchEngine] (const httplib::Request& req, httplib::Response& res) {
        if (req.has_param("q")) {
            std::string query = req.get_param_value("q");
            std::string result = searchEngine.search(query);
            res.set_content(result, "text/plain");
        }
        else {
            res.status = 400;
            res.set_content("Missing query parameter 'q'", "text/plain");
        }
    });

    std::cout << "Running on http://localhost:8080/search?q=<query>\n";

    server.listen("localhost", 8080);
}

