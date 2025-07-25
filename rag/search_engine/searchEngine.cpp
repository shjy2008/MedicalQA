#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <chrono>
#include "searchEngine.h"


int SearchEngine::calculateCounter = 0;
std::chrono::steady_clock::duration SearchEngine::timeCounter;

//std::string indexPath = "./";
std::string indexPath = "/projects/sciences/computing/sheju347/MedicalQA/rag/search_engine/";

void SearchEngine::load() {
	this->loadWords(); // load word postings index from disk
	this->loadDocLengths();
	this->loadDocOffsetTable();

	wordPostingsFile.open(indexPath + "index_postings.bin");
}

void SearchEngine::loadDocLengths() {
	std::ifstream docLengthsFile;
	docLengthsFile.open(indexPath + "index_docLengths.bin");

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

void SearchEngine::loadWords() {
	std::ifstream wordsFile;
	wordsFile.open(indexPath + "index_words.bin");

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

void SearchEngine::loadDocOffsetTable()  {
	std::ifstream docOffsetTableFile;
	docOffsetTableFile.open(indexPath + "index_docOffsetTable.bin");

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

std::pair<std::string, std::string> SearchEngine::getDocData(uint32_t docId) {
	DocumentOffset offsetData = this->docOffsetTable[docId - 1];

	std::ifstream documentsFile;
	documentsFile.open(indexPath + "index_documents.bin");
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

std::pair<std::vector<Posting>, float> SearchEngine::getWordPostings(const std::string& word) {
	std::vector<Posting> postings;

	std::unordered_map<std::string, WordData>::iterator wordDataItr = this->wordToWordData.find(word);
	if (wordDataItr == this->wordToWordData.end()) {
		return std::pair<std::vector<Posting>, float>(postings, 0); // Can't find the word, return empty vector
	}

	WordData wordData = wordDataItr->second;
	uint32_t pos = wordData.postingsPos;
	uint32_t docCount = wordData.postingsDocCount;
	float impactScore = wordData.impactScore;

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

	return std::pair<std::vector<Posting>, float>(postings, impactScore);
}

inline SearchResult SearchEngine::calculateDocScore(std::vector<std::vector<Posting> >& vecPostingsLists, 
						std::unordered_map<uint32_t, uint32_t>& wordIndexToPostingsProgress,
						std::priority_queue<Cursor, std::vector<Cursor>, std::greater<Cursor> >& cursorMinHeap) {
	float currentScore = 0.0f;
	uint32_t currentDocId = 0;
	SearchResult result = SearchResult(currentDocId, currentScore);
	uint32_t i = 0;
	while (cursorMinHeap.size() > 0) {
		Cursor cursor = cursorMinHeap.top();
		uint32_t wordIndex = cursor.wordIndex;
		std::vector<Posting>& postings = vecPostingsLists[wordIndex];
		
		Posting posting = postings[wordIndexToPostingsProgress[wordIndex]];

		uint32_t docCountContainWord = postings.size();
		float idf = Utils::getIDF(docCountContainWord, this->totalDocuments);

		if (i == 0 || posting.docId == currentDocId) {
			if (i == 0) {
				currentDocId = posting.docId;
			}
			uint32_t docLength = this->docLengthTable[currentDocId - 1];
			float score = Utils::getRankingScore(posting.tf, docLength, idf, this->averageDocumentLength);
			currentScore += score;

			calculateCounter += 1;

			wordIndexToPostingsProgress[wordIndex] += 1; // Advance the progress

			cursorMinHeap.pop();

			if (wordIndexToPostingsProgress[wordIndex] < postings.size()) {
				cursorMinHeap.push(Cursor(wordIndex, postings[wordIndexToPostingsProgress[wordIndex]].docId));
			}
		}
		else {
			break;
		}

		++i;
	}

	result = SearchResult(currentDocId, currentScore);
	return result;
}

std::vector<SearchResult> SearchEngine::getSortedRelevantDocuments(const std::string& query, size_t topK) {
	std::vector<std::string> words = Utils::extractWords(query);

	std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult> > resultMinHeap;

	// DAAT
	std::vector<std::vector<Posting> > vecPostingsLists; // [Postings, ...]
	std::unordered_map<uint32_t, float> wordIndexToImpactScores; // wordIndex -> impactScore
	std::unordered_map<uint32_t, uint32_t> wordIndexToPostingsProgress; // wordIndex -> postingsProgress (from 0 to len(posting) - 1)
	
	// std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

	uint32_t wordIndex = 0;
	for (uint32_t i = 0; i < words.size(); ++i) {
		std::string word = words[i];
		// Stop words
		if (std::find(stopWords.begin(), stopWords.end(), word) != stopWords.end()) {
			continue;
		}
		std::pair<std::vector<Posting>, float> postingsAndImpactScore = this->getWordPostings(word);
		if (postingsAndImpactScore.first.size() == 0 || postingsAndImpactScore.second == 0) {
			continue;
		}
		vecPostingsLists.push_back(postingsAndImpactScore.first);
		wordIndexToImpactScores[wordIndex] = postingsAndImpactScore.second;
		wordIndexToPostingsProgress[wordIndex] = 0;
		++wordIndex;

		// std::cout << "word:" << word << " postings size: " << postingsAndImpactScore.first.size() << std::endl;
	}

	// std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	// this->timeCounter += time_end - time_begin;	

	// WAND
	float minScoreOfHeap = 0.0f;

	// Use a min-heap to store the cursor of each word's postings
	std::priority_queue<Cursor, std::vector<Cursor>, std::greater<Cursor> > cursorMinHeap; 
	for (uint32_t i = 0; i < vecPostingsLists.size(); ++i) {
		cursorMinHeap.push(Cursor(i, vecPostingsLists[i][0].docId));
	}

	while (cursorMinHeap.size() > 0) {
		bool allFinished = false;

		if (resultMinHeap.size() < topK) {
			SearchResult ret = this->calculateDocScore(vecPostingsLists, wordIndexToPostingsProgress, cursorMinHeap);
			resultMinHeap.push(ret);
			minScoreOfHeap = resultMinHeap.top().score;
		}
		else {
			float currentImpactScore = 0.0f;
			
			uint32_t firstDocId = 0;
			uint32_t i = 0;
			std::priority_queue<Cursor, std::vector<Cursor>, std::greater<Cursor>> cursorMinHeapCopy = cursorMinHeap;
			while (cursorMinHeapCopy.size() > 0) {
				Cursor curosr = cursorMinHeapCopy.top();
				cursorMinHeapCopy.pop();
				uint32_t wordIndex = curosr.wordIndex;
				std::vector<Posting>& postings = vecPostingsLists[wordIndex]; // Notice that here need to use & reference, or will copy postings

				Posting posting = postings[wordIndexToPostingsProgress[wordIndex]];
				
				if (i == 0) {
					firstDocId = posting.docId;
				}

				float impactScore = wordIndexToImpactScores[wordIndex];
				currentImpactScore += impactScore;
				if (currentImpactScore > minScoreOfHeap) {
					if (posting.docId != firstDocId) { // d_p != d_0

						// Advance all lists to d >= d_p
						for (uint32_t j = 0; j < i; ++j) {
							Cursor cursor_j = cursorMinHeap.top();
							uint32_t wordIndex_j = cursor_j.wordIndex;
							std::vector<Posting>& postings_j = vecPostingsLists[wordIndex_j];

							cursorMinHeap.pop();

							// Only if the last posting > d_p, progress the cursor. Otherwise, the postings of this word are all done
							if (postings_j[postings_j.size() - 1].docId >= posting.docId) {
								// Replaced by Straddle Linear Search
								// while (true) {
								// 	if (postings_j[wordIndexToPostingsProgress[wordIndex_j]].docId >= posting.docId) {
								// 		// std::cout << wordIndex << " " << postings.size() << " " << wordIndexToPostingsProgress[wordIndex] << " " << postings_j[wordIndexToPostingsProgress[wordIndex_j]].docId << ">=" << posting.docId << std::endl;
								// 		break;
								// 	}

								// 	wordIndexToPostingsProgress[wordIndex_j] += 1;
								// }

								// std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
								
								// Straddle linear search
								uint32_t currentIndex = wordIndexToPostingsProgress[wordIndex_j];
								uint32_t searchDocId = posting.docId;
								uint32_t newProgress;
								if (postings_j[currentIndex].docId >= searchDocId) {
									newProgress = currentIndex;
								}
								else {
									// Straddle linear search
									uint32_t left = currentIndex;
									uint32_t right = 0;
									uint32_t jump = 2;
									while (true) {
										right = left + jump;
										if (right > postings_j.size() - 1) {
											right = postings_j.size() - 1;
										}
										if (postings_j[right].docId >= searchDocId) {
											break;
										}

										jump <<= 1; //jump *= 2;
										left = right;
									}

									// binary search
									while (true) {
										uint32_t gap = right - left;
										uint32_t middle = left + gap / 2;
										if (postings_j[middle].docId < searchDocId) {
											left = middle;
										}
										else {
											right = middle;
										}
										if (left == right - 1) {
											newProgress = right;
											break;
										}
									}
								}
								wordIndexToPostingsProgress[wordIndex_j] = newProgress;

								// std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
								// this->timeCounter += time_end - time_begin;	

								cursorMinHeap.push(Cursor(wordIndex_j, postings_j[wordIndexToPostingsProgress[wordIndex_j]].docId));
							}

						}

					}
					else { // d_p == d_0
						SearchResult ret = this->calculateDocScore(vecPostingsLists, wordIndexToPostingsProgress, cursorMinHeap);
						if (ret.score > minScoreOfHeap) {
							resultMinHeap.pop();
							resultMinHeap.push(ret);
							minScoreOfHeap = resultMinHeap.top().score;
						}
					}
					break;
				}
				else {
					if (i >= cursorMinHeap.size() - 1) {
						allFinished = true;
						break;
					}
				}

				++i;
			}
		}

		if (allFinished) {
			break;
		}
	}

	std::vector<SearchResult> vecDocIdScore; // docId and score: [(docId1, score1), (docId2, score2), ...]
	while (!resultMinHeap.empty()) {
		vecDocIdScore.push_back(resultMinHeap.top());
		resultMinHeap.pop();
	}

	// Sort by score
	std::reverse(vecDocIdScore.begin(), vecDocIdScore.end());

	return vecDocIdScore;
}

void SearchEngine::run() {
	// std::string query = "rosenfield wall street unilateral representation";
	// std::string query = "in antipeptic activity";
	// std::string query = "the";
	// std::string query = "in";
	// std::string query = "In vitro studies about the antipeptic activity";
	// std::string query = "vitro studies antipeptic activity";
	// std::string query = "junior orthopaedic surgery resident completing carpal tunnel repair";
	// GBaker/MedQA-USMLE-4-options test index 0
	// std::string query = "junior orthopaedic";
	std::string query = "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?";
	// std::string query = "junior orthopaedic surgery resident completing carpal tunnel repair department chairman attending physician. During case, resident inadvertently cuts flexor tendon. tendon repaired complication. attending tells resident patient fine, need report minor complication harm patient, he want make patient worry unnecessarily. He tells resident leave this complication operative report. Which following correct next action resident take?";
	// std::string query = "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?";
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

std::string SearchEngine::search(const std::string& query) {
	size_t topK = 3;
	std::vector<SearchResult> vecDocIdScore = this->getSortedRelevantDocuments(query, topK);

	std::string ret;
	for (const auto& result : vecDocIdScore) {
		std::pair<std::string, std::string> docData = this->getDocData(result.docId);
		ret += docData.second;
		ret += "\n\n";
	}
	return ret;
}

// int main() {
	
// 	std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

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

// 	return 0;
// }
