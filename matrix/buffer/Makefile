CC			= g++
CPPFLAGS	= -std=c++11 -Wall -march=native -O3
LDFLAGS		= -lOpenCL
SRCS		= matrix-buffer.cpp
OBJS		= $(SRCS:.cpp=.o)
PROG		= matrix-buffer

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CPPFLAGS) $(OBJS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CPPFLAGS) $< -c -o $@

clean:
	rm $(OBJS) $(PROG)
