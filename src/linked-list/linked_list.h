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
    int size;
} LinkedList;

// Function declarations
LinkedList* createList();
void insertAtBeginning(LinkedList* list, int data);
void insertAtEnd(LinkedList* list, int data);
void deleteNode(LinkedList* list, int data);
void printList(LinkedList* list);
void freeList(LinkedList* list);

#endif // LINKED_LIST_H 