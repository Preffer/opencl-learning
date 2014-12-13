CC		= g++
CPPFLAGS	= -std=c++11 -Wall -march=native -O3
LDFLAGS		=
SRCS		= matrix-cpu.cpp
OBJS		= $(SRCS:.cpp=.o)
PROG		= matrix-cpu

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CPPFLAGS) $(OBJS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CPPFLAGS) $< -c -o $@

clean:
	rm $(OBJS) $(PROG)
