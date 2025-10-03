package com.dsa.stack;

public class StackDemo {

    static class Node{

        int data;
        Node next;

        Node(int data,Node next){
            this.data = data;
            this.next = next;
        }
    }

    static class Stack{

        Node head=null;
        public boolean isEmpty(){
            if(head == null){
                return true;
            }else{
                return false;
            }
        }

        public void push(int data){
            Node newNode = new Node(data,null);
            if(head == null){
                head = newNode;
            }else{
                newNode.next = head;
                head = newNode;
            }
        }

        public int peek(){
            if(head == null){
                return -1;
            }else {
                return head.data;
            }
        }

        public int pop(){
            int data = 0;
            if(head == null){
                return -1;
            }else{
                data = head.data;
                head = head.next;
                return data;
            }
        }
    }

    public static void main(String[] arg){

        Stack stack=new Stack();
        stack.push(10);
        stack.push(20);
        stack.push(30);
        stack.push(40);
        stack.push(50);
        //System.out.println(stack.peek());
        while(!stack.isEmpty()) {
            System.out.print(" "+ stack.peek());
            stack.pop();
        }
    }
}
