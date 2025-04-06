#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#include <stdio.h>
#include <stdlib.h>

// Structure for a node in the linked list
typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Structure for the linked list
typedef struct {
    Node* head;
    Node* tail;  // Add tail pointer for O(1) append
    int size;
} LinkedList;

// Function declarations
LinkedList* createList();
void insertAtBeginning(LinkedList* list, int data);
void insertAtEnd(LinkedList* list, int data);
void deleteFirstOccurrence(LinkedList* list, int data);  // Renamed to be explicit
void printList(LinkedList* list);
void freeList(LinkedList* list);

#endif // LINKED_LIST_H 