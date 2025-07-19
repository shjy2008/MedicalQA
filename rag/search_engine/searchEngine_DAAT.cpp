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
#include "utils.h"

// Index files:
// 1. index_docLengths.bin: Document lengths for calculating scores. 4 bytes uint32_t each document length
// 2. index_docOffsetTable.bin: For each document length,  offset(8 byte) + docNoLength(1 byte) + documentLength(how many bytes, not words)(2 bytes)
// 3. index_documents.bin: For each document, docNo(str) + document(str)
// 4. index_words.bin: Words and their postings index, for seeking and reading word postings. Stored as 4 bytes word count + (wordLength(uint8_t), word, pos(uint32_t), docCount(uint32_t))
// 5. index_wordPostings.bin: Word postings file, stored as (docId1, tf1, docId2, tf2, ...) each 4 bytes

class SearchEngine {

private:
	std::ifstream wordPostingsFile; // Word postings file

	uint32_t totalDocuments; // number of documents in total, initialize after loading index_docLengths.bin
	float averageDocumentLength; // Average length of all the documents, used for BM25
	std::vector<uint32_t> docLengthTable; // docId -> documentLength

	// word -> (pos, docCount, impactScore)
	// -- pos: how many documents before the word's first document
	// -- docCount: how many documents the word appears in
	// -- impactScore: max impact score of a word for all documents
	std::unordered_map<std::string, WordData> wordToWordData;

	// Document offset table
	std::vector<DocumentOffset> docOffsetTable;

public:
	SearchEngine() {
	}

	void load() {
		this->loadWords(); // load word postings index from disk
		this->loadDocLengths();
		this->loadDocOffsetTable();

		wordPostingsFile.open("index_postings.bin");
	}

	// Load document lengths and get: 1. totalDocuments 2. average document length 3. docLengthTable (Used for BM25)
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
		uint64_t totalLength = 0;
		while (pointer < buffer + fileSize)
		{
			length = *reinterpret_cast<uint32_t*>(pointer);
			pointer += 4;

			++docId;

			this->docLengthTable.push_back(length);
			totalLength += length;
		}
		
		delete[] buffer;

		this->totalDocuments = docId;
		this->averageDocumentLength = totalLength / (float)this->totalDocuments;

		std::cout << "Finish loading document lengths" << std::endl;
	}

	// Load words and get the postings offset (this->wordToWordData)
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

			float impactScore = *reinterpret_cast<float*>(pointer);
			pointer += 4;

			this->wordToWordData[word] = WordData(pos, docCount, impactScore);
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
			this->docOffsetTable.push_back(DocumentOffset(offset, docNoLength, documentLength));
		}

		delete[] buffer;

		std::cout << "Finish loading document offset table: " << docId << std::endl;
	}

	// Read this->docOffsetTable and get <docNo and document> with docId
	std::pair<std::string, std::string> getDocData(uint32_t docId) {
		DocumentOffset offsetData = this->docOffsetTable[docId - 1];

		std::ifstream documentsFile;
		documentsFile.open("index_documents.bin");
		documentsFile.seekg(offsetData.offset, std::ifstream::beg);

		char* buffer = new char[offsetData.docNoLength];
		documentsFile.read(buffer, offsetData.docNoLength);
		std::string docNo(buffer, offsetData.docNoLength);
		delete[] buffer;

		buffer = new char[offsetData.documentLength];
		documentsFile.read(buffer, offsetData.documentLength);
		std::string document(buffer, offsetData.documentLength);
		delete[] buffer;

		return std::pair<std::string, std::string>(docNo, document);
	}

	// Get word postings. input: word
	// return: [(docId1, tf1), (docId2, tf2), ...], e.g. [(2, 3), (3, 6), ...]
	std::vector<Posting> getWordPostings(const std::string& word) {
		std::vector<Posting> postings;

		std::unordered_map<std::string, WordData>::iterator wordDataItr = this->wordToWordData.find(word);
		if (wordDataItr == this->wordToWordData.end()) {
			return postings; // Can't find the word, return empty vector
		}

		WordData postingsIndexPair = wordDataItr->second;
		uint32_t pos = postingsIndexPair.postingsPos;
		uint32_t docCount = postingsIndexPair.postingsDocCount;

		// Seek and read wordPostings.bin to find the postings(docId and tf) of this word
		wordPostingsFile.seekg(sizeof(uint32_t) * pos * 2, std::ifstream::beg); // * 2 because every doc has docId and term frequency

		// Optimization: batch reading instead of reading docId and tf one by one
		postings.resize(docCount);
		wordPostingsFile.read(reinterpret_cast<char*>(postings.data()), docCount * sizeof(Posting));

		// for (uint32_t i = 0; i < docCount; ++i) {
		// 	uint32_t docId = 0;
		// 	uint32_t tf = 0;
		// 	wordPostingsFile.read((char*)&docId, 4);
		// 	wordPostingsFile.read((char*)&tf, 4);

		// 	postings.push_back(Posting(docId, tf));
		// }

		return postings;
	}

	// input: query (multiple words) e.g. italy commercial
	// output: a list of sorted docId and score. e.g. [(1, 2.5), (10, 2.1), ...]
	std::vector<SearchResult> getSortedRelevantDocuments(const std::string& query) {
		std::vector<std::string> words = Utils::extractWords(query);

		// DAAT
		std::vector<std::pair<uint32_t, std::vector<Posting> > > vecPostingsLists; // [(wordIndex, Posting), ...]
		std::unordered_map<uint32_t, uint32_t> wordIndexToPostingsProgress; // wordIndex -> postingsProgress (from 0 to len(posting) - 1)
		
		uint32_t wordIndex = 0;
		for (uint32_t i = 0; i < words.size(); ++i) {
			std::string word = words[i];
			// Stop words
			if (std::find(stopWords.begin(), stopWords.end(), word) != stopWords.end()) {
				continue;
			}
			std::vector<Posting> postings = this->getWordPostings(word);
			vecPostingsLists.push_back(std::pair<uint32_t, std::vector<Posting> >(wordIndex, postings));
			wordIndexToPostingsProgress[wordIndex] = 0;
			++wordIndex;
		}

		// WAND
		std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult> > minHeap;
		const int topK = 10;
		float minScoreOfHeap = 0.0f;
		while (true) {
			// Sort the postings lists on increasing current docId
			std::sort(vecPostingsLists.begin(), vecPostingsLists.end(), [&](const auto& a, const auto& b) {
				bool aFinished = wordIndexToPostingsProgress[a.first] >= a.second.size();
				bool bFinished = wordIndexToPostingsProgress[b.first] >= b.second.size();
				if (aFinished && !bFinished) {
					return false;
				}
				else if (!aFinished && bFinished) {
					return true;
				}
				else if (aFinished && bFinished) {
					return a.first < b.first;
				}
				else {
					return a.second[wordIndexToPostingsProgress[a.first]].docId < b.second[wordIndexToPostingsProgress[b.first]].docId;
				}
			});

			float currentScore = 0.0f;
			uint32_t currentDocId = 0;
			bool allFinished = false;
			for (uint32_t i = 0; i < vecPostingsLists.size(); ++i) {
				std::vector<Posting> postings = vecPostingsLists[i].second;
				if (wordIndexToPostingsProgress[vecPostingsLists[i].first] >= postings.size()) {
					allFinished = true;
					break;
				}
				Posting posting = postings[wordIndexToPostingsProgress[vecPostingsLists[i].first]];
				uint32_t docCountContainWord = postings.size();
				float idf = Utils::getIDF(docCountContainWord, this->totalDocuments);

				if (i == 0 || posting.docId == currentDocId) {
					if (i == 0) {
						currentDocId = posting.docId;
					}
					uint32_t docLength = this->docLengthTable[currentDocId - 1];
					float score = Utils::getRankingScore(posting.tf, docLength, idf, this->averageDocumentLength);
					currentScore += score;

					wordIndexToPostingsProgress[vecPostingsLists[i].first] += 1; // Advance the progress
				}
				else {
					break;
				}
			}

			if (allFinished) {
				break;
			}

			// Add to the heap
			if (minHeap.size() < topK || currentScore > minScoreOfHeap) {
				if (minHeap.size() >= topK) {
					minHeap.pop();
				}
				minHeap.push(SearchResult(currentDocId, currentScore));
				minScoreOfHeap = minHeap.top().score;
			}
		}

		// TAAT
		// std::unordered_map<uint32_t, float> mapDocIdScore;
		// for (std::vector<std::string>::iterator itrWords = words.begin(); itrWords != words.end(); ++itrWords) {
		// 	std::string word = *itrWords;
		// 	for (size_t i = 0; i < word.length(); ++i)
		// 		word[i] = std::tolower(word[i]);

		// 	// Stop words
		// 	if (std::find(stopWords.begin(), stopWords.end(), word) != stopWords.end()) {
		// 		continue;
		// 	}

		// 	std::vector<Posting> postings = this->getWordPostings(word);
		// 	uint32_t docCountContainWord = postings.size();
		// 	float idf = Utils::getIDF(docCountContainWord, this->totalDocuments);

		// 	//std::cout << postings.size() << std::endl;
		// 	for (size_t i = 0; i < postings.size(); ++i) {
		// 		uint32_t docId = postings[i].docId; // docId (1, 2, 3, ...)
		// 		uint32_t tf_td = postings[i].tf; // term frequency in doc
	
		// 		// std::cout << docId << " " << tf_td << std::endl;

		// 		uint32_t docLength = this->docLengthTable[docId - 1];

		// 		float score = Utils::getRankingScore(tf_td, docLength, idf, this->averageDocumentLength);

		// 		// std::cout << docLength << " " << score << std::endl;

		// 		// Add score to mapDocIdScore
		// 		if (score > 0) {
		// 			std::unordered_map<uint32_t, float>::iterator itrMapDocIdScore = mapDocIdScore.find(docId);
		// 			if (itrMapDocIdScore != mapDocIdScore.end()) {
		// 				itrMapDocIdScore->second += score;
		// 			}
		// 			else {
		// 				mapDocIdScore[docId] = score;
		// 			}
		// 		}
		// 	}	
		// }

		// const int topK = 10;
		// for (std::unordered_map<uint32_t, float>::iterator itrMapDocIdScore = mapDocIdScore.begin(); itrMapDocIdScore != mapDocIdScore.end(); ++itrMapDocIdScore) {
		// 	if (minHeap.size() < topK) {
		// 		minHeap.push(SearchResult(itrMapDocIdScore->first, itrMapDocIdScore->second));
		// 	}
		// 	else {
		// 		if (itrMapDocIdScore->second > minHeap.top().score) {
		// 			minHeap.pop();
		// 			minHeap.push(SearchResult(itrMapDocIdScore->first, itrMapDocIdScore->second));
		// 		}
		// 	}
		// }

		std::vector<SearchResult> vecDocIdScore; // docId and score: [(docId1, score1), (docId2, score2), ...]
		while (!minHeap.empty()) {
			vecDocIdScore.push_back(minHeap.top());
			minHeap.pop();
		}

		// Sort by score
		std::reverse(vecDocIdScore.begin(), vecDocIdScore.end());

		return vecDocIdScore;
	}


	void run() {
		// std::string query = "rosenfield wall street unilateral representation";
		// std::string query = "in antipeptic activity";
		// std::string query = "the";
		// std::string query = "in";
		// std::string query = "In vitro studies about the antipeptic activity";
		// std::string query = "vitro studies antipeptic activity";
		// std::string query = "junior orthopaedic surgery resident completing carpal tunnel repair";
		// GBaker/MedQA-USMLE-4-options test index 0
		std::string query = "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?";
		// std::string query = "junior orthopaedic surgery resident completing carpal tunnel repair department chairman attending physician. During case, resident inadvertently cuts flexor tendon. tendon repaired complication. attending tells resident patient fine, need report minor complication harm patient, he want make patient worry unnecessarily. He tells resident leave this complication operative report. Which following correct next action resident take?";
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
