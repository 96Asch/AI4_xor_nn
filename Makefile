CC = g++

CompileParms = -g -Wall -Wextra -pedantic -O2

OBJS = nnskelet.o

DEPS =

EXEC = AI4

OS1: clean build
	
build: $(OBJS)
	$(CC) $(OBJS) -o $(EXEC)

%.o: %.c $(DEPS)
	$(CC) $(CompileParms) -o $@ -c $<

clean:
	$(RM) $(OBJS) $(EXEC)

c:
	$(RM) $(OBJS)

e:
	$(RM) $(EXEC)
