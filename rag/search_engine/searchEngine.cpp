#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

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

// There are four .bin index files:
// 1. index_docLengths.bin: Document lengths for calculating scores. 4 bytes uint32_t each document length
// 2. index_docNo.bin: DOCNO file, for showing DOCNO after retrieving docId. String splited by \0 (docNo1 \0 docNo2 \0 ...)
// 3. index_words.bin: Words and their postings index, for seeking and reading word postings. Stored as 4 bytes word count + (wordLength(uint8_t), word, pos(uint32_t), docCount(uint32_t))
// 4. index_wordPostings.bin: Word postings file, stored as (docId1, tf1, docId2, tf2, ...) each 4 bytes

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

	// DOCNO list ["WSJ870323-0139", ...]
	std::vector<std::string> vecDocNo;

public:
	SearchEngine() {
	}

	void load() {
		this->loadWords(); // load word postings index from disk
		this->loadDocNo(); // load DOCNO to a string list
		this->loadDocLengths();

		wordPostingsFile.open("index_wordPostings.bin");
	}

	// Load document lengths and get: 1. totalDocuments 2. average document length 3. docIdToLength (Used for BM25)
	void loadDocLengths() {
		std::ifstream docLengthsFile;
		docLengthsFile.open("index_docLengths.bin");

		docLengthsFile.seekg(0, std::fstream::end);
		long fileSize = docLengthsFile.tellg();
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
		long fileSize = wordsFile.tellg();
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
	}

	// Load DocNo.bin and push_back to this->vecDocNo
	void loadDocNo() {
		std::ifstream docNoFile; 
		docNoFile.open("index_docNo.bin");

		// Get the file size
		docNoFile.seekg(0, std::ifstream::end);
		long fileSize = docNoFile.tellg();
		docNoFile.seekg(0, std::ifstream::beg);

		// Create a buffer and load the entire file
		char* buffer = new char[fileSize];
		docNoFile.read(buffer, fileSize);

		docNoFile.close();

		char* pointer = buffer;

		while (pointer < buffer + fileSize) {
			std::string docNo(pointer);
			this->vecDocNo.push_back(docNo);
			pointer += (docNo.size() + 1); // +1 for the '\0' between pointers
		}

		delete[] buffer; // Release the buffer
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
	std::vector<std::pair<uint32_t, float> > getSortedRelevantDocuments(const std::string& query) {
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

			for (size_t i = 0; i < postings.size(); ++i) {
				uint32_t docId = postings[i].first; // docId (1, 2, 3, ...)
				uint32_t tf_td = postings[i].second; // term frequency in doc
	
				uint32_t docLength = this->getDocumentLength(docId);

				float score = this->getRankingScore(tf_td, docLength, idf);

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

		std::vector<std::pair<uint32_t, float> > vecDocIdScore; // docId and score: [(docId1, score1), (docId2, score2), ...]
		for (std::unordered_map<uint32_t, float>::iterator itrMapDocIdScore = mapDocIdScore.begin(); itrMapDocIdScore != mapDocIdScore.end(); ++itrMapDocIdScore) {
			vecDocIdScore.push_back(std::pair<uint32_t, float>(itrMapDocIdScore->first, itrMapDocIdScore->second));			
		}

		// Sort by score
		std::sort(vecDocIdScore.begin(), vecDocIdScore.end(), sortScoreCompare);

		return vecDocIdScore;
	}


	void run() {
		// std::string query = "rosenfield wall street unilateral representation";
		// std::string query = "hello";
		std::string query;
		std::getline(std::cin, query);
		std::vector<std::pair<uint32_t, float> > vecDocIdScore = this->getSortedRelevantDocuments(query);

		// Print the sorted list of docNo and score
		for (size_t i = 0; i < vecDocIdScore.size(); ++i) {
			uint32_t docId = vecDocIdScore[i].first;
			std::string docNo = this->vecDocNo[docId - 1];
			float score = vecDocIdScore[i].second;

			std::cout << docNo << " " << score << std::endl;
		}

		this->wordPostingsFile.close();
	}
};

int main() {
	
	// std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

	SearchEngine engine;
	engine.load();
	engine.run();

	// std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();

	// std::cout << "Time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count() << "ms" << std::endl;

	return 0;
}
