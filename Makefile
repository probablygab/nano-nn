# General purpose Makefile v3.0 - by Gab

# Set default target as all
.DEFAULT_GOAL := all

# Directories
MAIN_SRCDIR = src
BINDIR = bin
OBJDIR = obj
MAIN_INCDIR = include
LIBDIR = lib

# Compiler
CC = g++

# Output
EXEC = nano-nn

# Change .cpp to .o
SRC = $(wildcard $(MAIN_SRCDIR)/*.cpp)
OBJ = $(addprefix $(OBJDIR)/,$(notdir $(SRC:.cpp=.o)))

# Get all inc dirs
MULTI_INCDIR = $(wildcard $(MAIN_INCDIR)/*/)
ALL_INCDIR = -I $(MAIN_INCDIR) $(addprefix -I , $(MULTI_INCDIR))

# Flags
CFLAGS = $(ALL_INCDIR) -Wall -Wextra -pedantic -std=c++17 -fopenmp -O3 -march=native -mavx2 -mfma
DBGFLAGS = -g -fno-inline
LFLAGS = -L $(LIBDIR) -fopenmp

ifeq ($(OS),Windows_NT)
LFLAGS += -l:raylib.dll
else
LFLAGS += -l:libraylib.a -ldl
endif

ifeq ($(DEBUG),YES)
CFLAGS := $(CFLAGS) $(DBGFLAGS)
endif

# Ignore these files
.PHONY : compile all run clean valgrind

# Compile source to outputs .o 
compile: $(OBJ)

$(OBJDIR)/%.o: $(MAIN_SRCDIR)/%.cpp
	$(CC) -c $(CFLAGS) $< -o $@

# Link everything together
all: compile
	$(CC) -o $(BINDIR)/$(EXEC) $(OBJDIR)/*.o $(LFLAGS)

# Run the program
run:
	(cd $(BINDIR) && ./$(EXEC) $(ARGS))

# Delete the program and build files
clean:
	rm -f $(BINDIR)/$(EXEC)
	rm -f $(BINDIR)/$(EXEC).exe
	rm -f $(OBJDIR)/*.o
	
# Run valgrind to search for memory leaks
valgrind:
	(cd $(BINDIR) && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./$(EXEC) $(ARGS))