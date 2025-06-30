#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <cstdint>

// Index files:
// 1. index_docLengths.bin: Document lengths for calculating scores. 4 bytes uint32_t each document length
// 2. index_docOffsetTable.bin: For each document length,  offset(8 byte) + docNoLength(1 byte) + documentLength(how many bytes, not words)(2 bytes)
// 3. index_documents.bin: For each document, docNo(str) + document(str)
// 4. index_words.bin: Words and their postings index, for seeking and reading word postings. Stored as 4 bytes word count + (wordLength(uint8_t), word, pos(uint32_t), docCount(uint32_t))
// 5. index_wordPostings.bin: Word postings file, stored as (docId1, tf1, docId2, tf2, ...) each 4 bytes

// Extract words from a text string
std::vector<std::string> extractWords(const std::string& text) {
	std::vector<std::string> words;

	std::string word;
	for (size_t i = 0; i < text.size(); ++i){
		if (std::isalpha(text[i]))
			word += std::tolower(text[i]);
		else if (std::isdigit(text[i]))
			word += text[i];
		// else if (text[i] == '-') { // Some words have '-', such as "well-being"
		// 	if (i > 0 && std::isalnum(text[i - 1])) { // Ignore words starting with '-'
		// 		word += text[i];
		// 	}
		// }
		else {
			if (word.length() > 0) {
				if (word.length() > 255) { // Length is stored in uint8_t, so truncate the word if its length > 255
					word = word.substr(0, 255);
				}
				words.push_back(word);
			}
			word = "";
		}
	}
	if (word.length() > 0)
		words.push_back(word);

	return words;
}

// Used for sorting the docId and its relevance score
bool sortScoreCompare(const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
	return a.second > b.second;
}

struct SearchResult {
	uint32_t docId;
	float score;

	SearchResult(uint32_t docId, float score) : docId(docId), score(score) {}

	bool operator > (const SearchResult& other) const {
		return score > other.score; // min-heap based on score
	}
};

class SearchEngine {

private:
	std::ifstream wordPostingsFile; // Word postings file

	uint32_t totalDocuments; // number of documents in total, initialize after loading index_docLengths.bin
	float averageDocumentLength; // Average length of all the documents, used for BM25
	std::unordered_map<uint32_t, uint32_t> docIdToLength; // docId -> documentLength

	// word -> (pos, docCount)
	// -- pos: how many documents before the word's first document
	// -- docCount: how many documents the word appears in
	std::unordered_map<std::string, std::pair<uint32_t, uint32_t> > wordToPostingsIndex;

	// Document offset table
	std::unordered_map<uint32_t, std::tuple<uint64_t, uint8_t, uint16_t> > docOffsetTable;

public:
	SearchEngine() {
	}

	void load() {
		this->loadWords(); // load word postings index from disk
		this->loadDocLengths();
		this->loadDocOffsetTable();

		wordPostingsFile.open("index_postings.bin");
	}

	// Load document lengths and get: 1. totalDocuments 2. average document length 3. docIdToLength (Used for BM25)
	void loadDocLengths() {
		std::ifstream docLengthsFile;
		docLengthsFile.open("index_docLengths.bin");

		docLengthsFile.seekg(0, std::fstream::end);
		uint64_t fileSize = docLengthsFile.tellg();
		docLengthsFile.seekg(0, std::fstream::beg);

		char* buffer = new char[fileSize];
		docLengthsFile.read(buffer, fileSize);

		docLengthsFile.close();

		char* pointer = buffer;

		uint32_t length = 0;
		uint32_t docId = 0;
		uint32_t totalLength = 0;
		while (pointer < buffer + fileSize)
		{
			length = *reinterpret_cast<uint32_t*>(pointer);
			pointer += 4;

			++docId;

			this->docIdToLength[docId] = length;
			totalLength += length;
		}
		
		delete[] buffer;

		this->totalDocuments = docId;
		this->averageDocumentLength = (float)totalLength / this->totalDocuments;

		std::cout << "Finish loading document lengths" << std::endl;

		// int a = getDocumentLength(1);
		// int b = getDocumentLength(2);
		// int c = getDocumentLength(1000000);
		// int d = 0;
	}

	// Get document length(how many words in doc) with docId: 1, 2, 3, ... (read from the postings)
	uint32_t getDocumentLength(uint32_t docId) {
		std::unordered_map<uint32_t, uint32_t>::iterator itr = this->docIdToLength.find(docId);
		if (itr != this->docIdToLength.end()) {
			return itr->second;
		}
		return 0;
	}

	// Load words and get the postings offset (this->wordToPostingsIndex)
	void loadWords() {
		std::ifstream wordsFile;
		wordsFile.open("index_words.bin");

		// Get how many bytes the words.bin have
		wordsFile.seekg(0, std::ifstream::end);
		uint64_t fileSize = wordsFile.tellg();
		wordsFile.seekg(0, std::ifstream::beg);

		// Batch reading is faster than reading byte by byte
		char* buffer = new char[fileSize];
		wordsFile.read(buffer, fileSize);
		
		wordsFile.close();

		char* pointer = buffer;

		uint32_t wordCount = *reinterpret_cast<uint32_t*>(pointer);
		pointer += 4;

		for (uint32_t i = 0; i < wordCount; ++i) {
			uint8_t wordLength = *pointer; //*reinterpret_cast<uint8_t*>(pointer);
			++pointer;

			std::string word(pointer, wordLength);
			pointer += wordLength;

			uint32_t pos = *reinterpret_cast<uint32_t*>(pointer);
			pointer += 4;

			uint32_t docCount = *reinterpret_cast<uint32_t*>(pointer);
			pointer += 4;

			this->wordToPostingsIndex[word] = std::pair<uint32_t, uint32_t>(pos, docCount);
		}

		delete[] buffer;

		std::cout << "Finish loading words: " << wordCount << std::endl;
	}

	// Load index_docOffsetTable.bin and update this->docOffsetTable
	void loadDocOffsetTable() {
		std::ifstream docOffsetTableFile;
		docOffsetTableFile.open("index_docOffsetTable.bin");

		// Get how many bytes the "index_docOffsetTable.bin" have
		docOffsetTableFile.seekg(0, std::ifstream::end);
		uint64_t fileSize = docOffsetTableFile.tellg();
		docOffsetTableFile.seekg(0, std::ifstream::beg);

		// Batch reading is faster than reading byte-by-byte
		char* buffer = new char[fileSize];
		docOffsetTableFile.read(buffer, fileSize);

		docOffsetTableFile.close();

		char* pointer = buffer;
		uint32_t docId = 0;
		while (pointer < buffer + fileSize) {
			uint64_t offset = *reinterpret_cast<uint64_t*>(pointer);
			pointer += sizeof(offset);

			uint8_t docNoLength = *reinterpret_cast<uint8_t*>(pointer);
			pointer += sizeof(docNoLength);

			uint16_t documentLength = *reinterpret_cast<uint16_t*>(pointer);
			pointer += sizeof(documentLength);

			++docId;
			this->docOffsetTable[docId] = std::tuple<uint64_t, uint8_t, uint16_t>(offset, docNoLength, documentLength);
		}

		delete[] buffer;

		std::cout << "Finish loading document offset table" << std::endl;
	}

	// Read this->docOffsetTable and get <docNo and document> with docId
	std::pair<std::string, std::string> getDocData(uint32_t docId) {
		std::tuple<uint64_t, uint8_t, uint16_t> tableData = this->docOffsetTable[docId];
		uint64_t offset = std::get<0>(tableData);
		uint8_t docNoLength = std::get<1>(tableData);
		uint16_t documentLength = std::get<2>(tableData);

		std::ifstream documentsFile;
		documentsFile.open("index_documents.bin");
		documentsFile.seekg(offset, std::ifstream::beg);

		char* buffer = new char[docNoLength];
		documentsFile.read(buffer, docNoLength);
		std::string docNo(buffer, docNoLength);
		delete[] buffer;

		buffer = new char[documentLength];
		documentsFile.read(buffer, documentLength);
		std::string document(buffer, documentLength);
		delete[] buffer;

		return std::pair<std::string, std::string>(docNo, document);
	}

	// Get word postings. input: word
	// return: [(docId1, tf1), (docId2, tf2), ...], e.g. [(2, 3), (3, 6), ...]
	std::vector<std::pair<uint32_t, uint32_t> > getWordPostings(const std::string& word) {
		std::vector<std::pair<uint32_t, uint32_t> > postings;

		std::unordered_map<std::string, std::pair<uint32_t, uint32_t> >::iterator postingsIndexIt = this->wordToPostingsIndex.find(word);
		if (postingsIndexIt == this->wordToPostingsIndex.end()) {
			return postings; // Can't find the word, return empty vector
		}

		std::pair<uint32_t, uint32_t> postingsIndexPair = postingsIndexIt->second;
		uint32_t pos = postingsIndexPair.first;
		uint32_t docCount = postingsIndexPair.second;

		// Seek and read wordPostings.bin to find the postings(docId and tf) of this word
		wordPostingsFile.seekg(sizeof(uint32_t) * pos * 2, std::ifstream::beg); // * 2 because every doc has docId and term frequency
		for (uint32_t i = 0; i < docCount; ++i) {
			uint32_t docId = 0;
			uint32_t tf = 0;
			wordPostingsFile.read((char*)&docId, 4);
			wordPostingsFile.read((char*)&tf, 4);

			postings.push_back(std::pair<uint32_t, uint32_t>(docId, tf));
		}

		return postings;
	}

	// tf_td: number of the term appears in doc
	// docLength: how many words in the document
	// idf: inverted document frequency (calculated by total document and documents contain the word)
	float getRankingScore(uint32_t tf_td, uint32_t docLength, float idf) {
		// TF-IDF
		// float tf_td_normalized = (float)tf_td / docLength;
		// float idf = (float)this->totalDocuments / docCountContainWord;
		// return tf_td_normalized * idf;

		// BM25 - in the slides (but it will produce negative value when docCountContainWord > 1/2 totalDocuments, then the ranking is wrong)
		// float w_t = std::log2f((this->totalDocuments - docCountContainWord + 0.5f) / (docCountContainWord + 0.5f));
		// float k1 = 1.2f;
		// float k3 = 7;
		// float b = 0.75f;
		// float K = k1 * ((1 - b) + (b * docLength / this->averageDocumentLength));
		// float w_dt = w_t * ((k1 + 1) * tf_td / (K + tf_td)) * ((k3 + 1) * tf_tq / (k3 + tf_tq));
		// return w_dt;

		// Okapi BM25 https://en.wikipedia.org/wiki/Okapi_BM25
		float k1 = 1.2f;
		float b = 0.75f;
		float K = k1 * ((1 - b) + b * (docLength / this->averageDocumentLength));
		float score = idf * (tf_td * (k1 + 1) / (tf_td + K));
		return score;
	}

	// input: query (multiple words) e.g. italy commercial
	// output: a list of sorted docId and score. e.g. [(1, 2.5), (10, 2.1), ...]
	std::vector<SearchResult> getSortedRelevantDocuments(const std::string& query) {
		std::vector<std::string> words = extractWords(query);

		std::unordered_map<uint32_t, float> mapDocIdScore;
		for (std::vector<std::string>::iterator itrWords = words.begin(); itrWords != words.end(); ++itrWords) {
			std::string word = *itrWords;
			for (size_t i = 0; i < word.length(); ++i)
				word[i] = std::tolower(word[i]);

			std::vector<std::pair<uint32_t, uint32_t> > postings = this->getWordPostings(word);
			uint32_t docCountContainWord = postings.size();

			// Okapi BM25 https://en.wikipedia.org/wiki/Okapi_BM25
			float idf = std::log((this->totalDocuments - docCountContainWord + 0.5) / (docCountContainWord + 0.5) + 1); // Ensure positive

			//std::cout << postings.size() << std::endl;
			for (size_t i = 0; i < postings.size(); ++i) {
				uint32_t docId = postings[i].first; // docId (1, 2, 3, ...)
				uint32_t tf_td = postings[i].second; // term frequency in doc
	
				// std::cout << docId << " " << tf_td << std::endl;

				uint32_t docLength = this->getDocumentLength(docId);

				float score = this->getRankingScore(tf_td, docLength, idf);

				// std::cout << docLength << " " << score << std::endl;

				// Add score to mapDocIdScore
				if (score > 0) {
					std::unordered_map<uint32_t, float>::iterator itrMapDocIdScore = mapDocIdScore.find(docId);
					if (itrMapDocIdScore != mapDocIdScore.end()) {
						itrMapDocIdScore->second += score;
					}
					else {
						mapDocIdScore[docId] = score;
					}
				}
			}	
		}

		std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult> > minHeap; 
		const int topK = 10;
		for (std::unordered_map<uint32_t, float>::iterator itrMapDocIdScore = mapDocIdScore.begin(); itrMapDocIdScore != mapDocIdScore.end(); ++itrMapDocIdScore) {
			// filter out score < a threshold
			// if (itrMapDocIdScore->second < 5) {
			// 	continue;
			// }

			if (minHeap.size() < topK) {
				minHeap.push(SearchResult(itrMapDocIdScore->first, itrMapDocIdScore->second));
			}
			else {
				if (itrMapDocIdScore->second > minHeap.top().score) {
					minHeap.pop();
					minHeap.push(SearchResult(itrMapDocIdScore->first, itrMapDocIdScore->second));
				}
			}
		}

		std::vector<SearchResult> vecDocIdScore; // docId and score: [(docId1, score1), (docId2, score2), ...]
		while (!minHeap.empty()) {
			vecDocIdScore.push_back(minHeap.top());
			minHeap.pop();
		}

		// Sort by score
		// std::sort(vecDocIdScore.begin(), vecDocIdScore.end(), sortScoreCompare);
		std::reverse(vecDocIdScore.begin(), vecDocIdScore.end());

		return vecDocIdScore;
	}


	void run() {
		// std::string query = "rosenfield wall street unilateral representation";
		// std::string query = "in antipeptic activity";
		// std::string query = "the";
		// std::string query = "in";
		// std::string query = "In vitro studies about the antipeptic activity";
		// GBaker/MedQA-USMLE-4-options test index 0
		std::string query = "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?";
		// std::string query;
		// std::getline(std::cin, query);

		std::vector<SearchResult> vecDocIdScore = this->getSortedRelevantDocuments(query);

		std::vector<uint32_t> bestDocIdList;

		// Print the sorted list of docNo and score
		for (size_t i = 0; i < vecDocIdScore.size(); ++i) {
			uint32_t docId = vecDocIdScore[i].docId;
			float score = vecDocIdScore[i].score;

			std::cout << docId << "  " << score << std::endl;

			if (i < 10) {
				bestDocIdList.push_back(docId);
			}
		}

		this->wordPostingsFile.close();

		// Print the first two docId and document content
		for (size_t i = 0; i < bestDocIdList.size(); ++i) {
			std::pair<std::string, std::string> docData = this->getDocData(bestDocIdList[i]);
			std::string docNo = docData.first;
			std::string document = docData.second;

			std::cout << std::endl << docNo << std::endl;
			std::cout << std::endl << document << std::endl;
		}
	}
};

int main() {
	
	std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

	SearchEngine engine;
	engine.load();

	std::chrono::steady_clock::time_point time_loadFinished = std::chrono::steady_clock::now();
	std::cout << "Loading time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_loadFinished - time_begin).count() << "ms" << std::endl;

	engine.run();

	std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	std::cout << "Search time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_loadFinished).count() << "ms" << std::endl;

	return 0;
}
