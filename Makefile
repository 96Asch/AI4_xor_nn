CXX ?= g++

CXXFLAGS = -Wall -Wextra -pedantic -std=c++11 -g -effc++

OBJS = nnskelet.o

DEPS =

EXEC = AI4

OS1: clean build
	
build: $(OBJS)
	$(CXX) -g $(OBJS) -o $(EXEC)

%.o: %.c $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	$(RM) $(OBJS) $(EXEC)

c:
	$(RM) $(OBJS)
	$(RM) vgco*

e:
	$(RM) $(EXEC)
	$(RM) vgco*
