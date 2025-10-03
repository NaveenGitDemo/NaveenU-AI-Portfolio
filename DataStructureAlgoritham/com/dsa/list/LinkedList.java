package com.dsa.list;

class Node {
    public int data;
    public Node next;

    Node(int data , Node next){
        this.data = data;
        this.next = null;
    }

    @Override
    public String toString() {
        return "Node[" +
                "data=" + data +
                ", next=" + next +
                ']';
    }

}
public class LinkedList {


    Node head;// List starting point

    @Override
    public String toString() {
        Node n = head;
        String display = "";
        while (n.next != null) {
            display = display + n.data + ",";
            n = n.next;
        }
        display = display + n.data;
        return "[" + display + ']';
    }

    // add data to list
    public void add(int data){

       Node  node = new Node(data,null);
       // pointer for non empty list
        Node n = head;
       if(head == null){
           head = node ;
       }else{
           while(n.next != null){
               n = n.next;
           }
           n.next = node;
       }

    }

    // iteration of list element
    public void show(){

        if(head == null){
            System.out.println("List is Empty");
        }else{
            Node temp = head;
            while(temp != null){
                System.out.println(temp.data);
                temp = temp.next;
            }
        }
    }

    // size to get size of the list
    public int size(){
        int count = 0;
        if(head == null)
        return -1;

        Node n = head;
        while(n!=null){
            count+=1;
            n = n.next;
        }
        return count;
    }

    // reverse linked list
    public void revrseList(){
        int i =0; int size = size() - 1; int k = 0;
         // pointer for movement of node
        Node p;
        Node q;
        p = head;
        q = head;
        while(i < size){
                k = 0;
            while(k < size){
                q = q.next;
                k++;
            }
            int temp = q.data;
            q.data = p.data;
            p.data = temp;

            // pointer movement
            p = p.next;
            q = head;

            i++;
            size = size - 1 ;
        }
    }

    public static void main(String [] arg){

        LinkedList list = new LinkedList();
        list.add(10);
        list.add(20);
        list.add(30);
        list.add(40);
        list.show();
        System.out.println(list);
        System.out.println("size of list : "+list.size());
        list.revrseList();
        System.out.println("reverse list : "+list);
    }

}
