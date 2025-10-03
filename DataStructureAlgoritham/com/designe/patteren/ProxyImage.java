package com.designe.patteren;

public class ProxyImage implements Image{

    private RealImage realImage;
    private String fileName;

    public ProxyImage(String fileName) {

        this.fileName = fileName;
    }

    @Override
    public void display(){

        if(realImage == null){
            System.out.println("load file first time realImage null so call constructor do heavy operation.. ");
            realImage = new RealImage(fileName);
        }
            realImage.display();
    }
}
