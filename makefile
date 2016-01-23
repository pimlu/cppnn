OPT=
CXXFLAGS=$(OPT) -c -g -O2 -Wall -std=c++11
LDFLAGS=

SOURCES=$(shell find src -name "*.cpp")
OBJECTS:=$(SOURCES:src/%.cpp=build/%.o)
EXECUTABLE=cppnn

all: $(SOURCES) dist/$(EXECUTABLE)

dist/$(EXECUTABLE): $(OBJECTS) $(PARSEO)
	@mkdir -p dist
	$(CXX) $(OBJECTS) $(PARSEO) $(LDFLAGS) -o $@

build:
	mkdir -p build
	#find src/* -type d -print0 | sed 's/src/build/g' | xargs -r0 mkdir -p

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

clean: FORCE
	rm -rf build

rmdist: FORCE
	rm -rf dist/*

FORCE: