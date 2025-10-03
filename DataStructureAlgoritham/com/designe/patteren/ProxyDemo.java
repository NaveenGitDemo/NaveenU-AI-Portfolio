package com.designe.patteren;

public class ProxyDemo {

    public static void main(String[] arg){
        Image image = new ProxyImage("test.jpg");

        // Image will be loaded only once (lazy loading)
        image.display();  // Loads from disk + displays
        System.out.println("---");
        image.display();  // Uses cached object, no load
    }
}
