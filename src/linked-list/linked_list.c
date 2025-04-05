#include "linked_list.h"

LinkedList* createList() {
    LinkedList* list = (LinkedList*)malloc(sizeof(LinkedList));
    if (list != NULL) {
        list->head = NULL;
        list->tail = NULL;
        list->size = 0;
    }
    return list;
}

void insertAtBeginning(LinkedList* list, int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode != NULL) {
        newNode->data = data;
        newNode->next = list->head;
        list->head = newNode;
        if (list->tail == NULL) {
            list->tail = newNode;
        }
        list->size++;
    }
}

void insertAtEnd(LinkedList* list, int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode != NULL) {
        newNode->data = data;
        newNode->next = NULL;
        
        if (list->head == NULL) {
            list->head = newNode;
            list->tail = newNode;
        } else {
            list->tail->next = newNode;
            list->tail = newNode;
        }
        list->size++;
    }
}

void deleteFirstOccurrence(LinkedList* list, int data) {
    if (list->head == NULL) return;

    Node* current = list->head;
    Node* prev = NULL;

    // If head node itself holds the data to be deleted
    if (current != NULL && current->data == data) {
        list->head = current->next;
        if (list->tail == current) {
            list->tail = NULL;
        }
        free(current);
        list->size--;
        return;
    }

    // Search for the data to be deleted
    while (current != NULL && current->data != data) {
        prev = current;
        current = current->next;
    }

    // If data was not present in linked list
    if (current == NULL) return;

    // Unlink the node from linked list
    prev->next = current->next;
    if (list->tail == current) {
        list->tail = prev;
    }
    free(current);
    list->size--;
}

void printList(LinkedList* list) {
    Node* current = list->head;
    printf("List (size: %d): ", list->size);
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");
}

void freeList(LinkedList* list) {
    Node* current = list->head;
    while (current != NULL) {
        Node* next = current->next;
        free(current);
        current = next;
    }
    free(list);
} 