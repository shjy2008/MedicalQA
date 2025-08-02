#include <string>
#include "httplib.h"
#include "searchEngine.h"

std::string ip = "127.0.0.1"; // "localhost"
int port = 8080;
std::string endpoint = "search";

// int main() {
//     std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

// 	SearchEngine engine;
// 	engine.load();

// 	std::chrono::steady_clock::time_point time_loadFinished = std::chrono::steady_clock::now();
// 	std::cout << "Loading time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_loadFinished - time_begin).count() << "ms" << std::endl;

// 	engine.run();
// 	//std::string query = "vitro studies antipeptic activity";
// 	// std::string query = "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?";
// 	// std::cout << engine.search(query) << std::endl;

// 	std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
// 	std::cout << "Search time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_loadFinished).count() << "ms" << std::endl;

// 	std::cout << "calculate counter:" << engine.calculateCounter << std::endl;
// 	std::cout << "time counter:" << std::chrono::duration_cast<std::chrono::milliseconds>(engine.timeCounter).count() << std::endl;
// }

int main() {
    httplib::Server server;

    SearchEngine searchEngine;
    searchEngine.load();

    server.Get("/" + endpoint, [&searchEngine] (const httplib::Request& req, httplib::Response& res) {
        if (req.has_param("q")) {
            std::string result;
            std::string query = req.get_param_value("q");
            if (req.has_param("k")) {
                std::string topK_str = req.get_param_value("k");
                try {
                    size_t topK = std::stoul(topK_str);
                    result = searchEngine.search(query, topK);
                }
                catch (const std::exception& e) {
                    std::cerr << "Invalid value for k: " << topK_str << std::endl;
                }
            }
            else {
                result = searchEngine.search(query);
            }
            res.set_content(result, "text/plain");
        }
        else {
            res.status = 400;
            res.set_content("Missing query parameter 'q'", "text/plain");
        }
    });

    std::cout << "Running on http://" << ip << ":" << port << "/search?q=<query>\n";

    server.listen(ip, port);
}

