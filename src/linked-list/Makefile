CC=gcc
CFLAGS=-Wall -Wextra

all: linked_list

linked_list: main.c linked_list.c
	$(CC) $(CFLAGS) -o linked_list main.c linked_list.c

clean:
	rm -f linked_list 