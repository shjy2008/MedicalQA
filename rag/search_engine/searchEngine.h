#ifndef __SEARCH_ENGINE__
#define __SEARCH_ENGINE__

#include <cstdint>
#include <fstream>
#include <vector>
#include <queue>
#include "utils.h"

// Index files:
// 1. index_docLengths.bin: Document lengths for calculating scores. 4 bytes uint32_t each document length
// 2. index_docOffsetTable.bin: For each document length,  offset(8 byte) + docNoLength(1 byte) + documentLength(how many bytes, not words)(2 bytes)
// 3. index_documents.bin: For each document, docNo(str) + document(str)
// 4. index_words.bin: Words and their postings index, for seeking and reading word postings. Stored as 4 bytes word count + (wordLength(uint8_t), word, pos(uint32_t), docCount(uint32_t))
// 5. index_wordPostings.bin: Word postings file, stored as (docId1, tf1, docId2, tf2, ...) each 4 bytes

struct Cursor {
	uint32_t wordIndex;
	uint32_t currentDocId;

	Cursor(uint32_t wordIndex, uint32_t currentDocId) : wordIndex(wordIndex), currentDocId(currentDocId) {}

	bool operator > (const Cursor& other) const {
		return currentDocId > other.currentDocId;
	}
};

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
	// For performance debugging
	static int calculateCounter;
	static std::chrono::steady_clock::duration timeCounter;

	SearchEngine() {
	}

	void load();

	// Load document lengths and get: 1. totalDocuments 2. average document length 3. docLengthTable (Used for BM25)
	void loadDocLengths();

	// Load words and get the postings offset (this->wordToWordData)
	void loadWords();

	// Load index_docOffsetTable.bin and update this->docOffsetTable
	void loadDocOffsetTable();

	// Read this->docOffsetTable and get <docNo and document> with docId
	std::pair<std::string, std::string> getDocData(uint32_t docId);

	// Get word postings. input: word
	// return: [(docId1, tf1), (docId2, tf2), ...], e.g. [(2, 3), (3, 6), ...]
	std::pair<std::vector<Posting>, float> getWordPostings(const std::string& word);

	inline SearchResult calculateDocScore(std::vector<std::vector<Posting> >& vecPostingsLists, 
						std::vector<uint32_t>& vecPostingsProgress,
						std::priority_queue<Cursor, std::vector<Cursor>, std::greater<Cursor> >& cursorMinHeap);

	// input: query (multiple words) e.g. italy commercial
	// output: a list of sorted docId and score. e.g. [(1, 2.5), (10, 2.1), ...]
	std::vector<SearchResult> getSortedRelevantDocuments(const std::string& query, size_t topK = 10);

	void run();

    std::vector<SearchResult> search(const std::string& query, size_t topK = 10);
};


#endif