
#pragma once
#ifndef _Timer
#define _Timer

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <list>
#include <map>
#include <string>
#include <iostream>

using std::list;
using std::map;
using std::string;

class timer {
private:
  map<string, list<cudaEvent_t>*> lis;

public:

  timer() {
    lis = map<string, list<cudaEvent_t>*>();
  }

  ~timer() {
    for (auto iter = lis.rbegin(); iter != lis.rend(); iter++) {
      list<cudaEvent_t>* lis_e = iter->second;
      for (auto iter2 = lis_e->begin(); iter2 != lis_e->end(); iter2++) {
        cudaEventDestroy(*iter2);
      }
      delete lis_e;
    }
  }

  void newEvent(const char name[], cudaStream_t stream = 0) {
    string s = string(name);
    newEvent(s, stream);
  }

  void newEvent(string& name, cudaStream_t stream = 0) {
    using std::pair;

    auto lis_i = lis.find(name);
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    if (lis_i == lis.end()) {
      list<cudaEvent_t>* lis_e = new list<cudaEvent_t>();
      lis_e->push_back(event);
      lis.insert(pair<string, list<cudaEvent_t>*>(name, lis_e));
    }
    else { 
      list<cudaEvent_t>* lis_e = lis_i->second;
      lis_e->push_back(event);
    }
  }

  void dumpEvents() {
    cudaDeviceSynchronize();
    using std::cout;
    using std::endl;

    cout << "Timer:" << endl;

    for (auto iter = lis.rbegin(); iter != lis.rend(); iter++) {
      string name = iter->first;
      list<cudaEvent_t>* lis_e = iter->second;
      cudaEvent_t start = lis_e->front();
      lis_e->pop_front();

      for (auto iter2 = lis_e->begin(); iter2 != lis_e->end(); iter2++) {
        cudaEvent_t end = *iter2;
        float time;
        cudaEventElapsedTime(&time, start, end);
        cout << name << ": " << time << endl;
        cudaEventDestroy(end);
      }
      cudaEventDestroy(start);
      delete lis_e;
    }

    lis.clear();
    cout << endl;
  }

};

#endif