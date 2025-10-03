package com.dsa.queue;

public class QueueDemo {
        static class Node{
            int data;
            Node next;

            public Node(int data) {
                this.data=data;
                next=null;
            }
        }
        static class Queue{
            // methods need to implement related to queue operation
            Node front=null ; Node rear=null;
            public boolean isEmpty() {
                if(front == null && rear == null) {
                    return true;
                }else {
                    return false;
                }
            }
            public void offer( int data) {
                Node node=new Node(data);
                if(isEmpty()) {
                    front = rear = node ;
                    System.out.println("inserted element in queue "+data);
                }else {
                    rear.next=node;
                    rear=rear.next;
                    System.out.println("insrted element in queue "+rear.data);
                }
            }
            public int poll() {
                int data;
                if(isEmpty()) {
                    return -1;
                }else {
                    data=front.data;
                    front=front.next;
                    if(front == null) {
                        rear = null ;
                    }
                }
                return data;
            }

        }

        public static void main(String [] arg) {

            Queue queue=new Queue() ;
            queue.offer(10);
            queue.offer(20);
            queue.offer(30);
            queue.offer(40);
            queue.offer(50);

            while(!queue.isEmpty()) {
                System.out.println("remove and return element from queue  "+queue.poll());
            }
        }
    }
