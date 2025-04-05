#include "linked_list.h"

int main() {
    // Create a new linked list
    LinkedList* list = createList();

    // Insert some elements
    insertAtBeginning(list, 10);
    insertAtEnd(list, 20);
    insertAtBeginning(list, 5);
    insertAtEnd(list, 30);

    // Print the list
    printf("Initial list:\n");
    printList(list);

    // Delete an element
    printf("\nDeleting 20 from the list...\n");
    deleteNode(list, 20);
    printList(list);

    // Insert more elements
    printf("\nInserting more elements...\n");
    insertAtEnd(list, 40);
    insertAtBeginning(list, 1);
    printList(list);

    // Free the list
    freeList(list);
    return 0;
} 