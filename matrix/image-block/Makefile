CC			= g++
CPPFLAGS	= -std=c++11 -Wall -march=native -O3
LDFLAGS		= -lOpenCL
SRCS		= matrix-image.cpp
OBJS		= $(SRCS:.cpp=.o)
PROG		= matrix-image

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CPPFLAGS) $(OBJS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CPPFLAGS) $< -c -o $@

clean:
	rm $(OBJS) $(PROG)
