package com.designe.patteren;

public class RealImage implements Image{

    private String fileName;

    RealImage(String fileName){
        this.fileName = fileName;
        loadFileFromDisc(fileName); // heavy work load here
    }

    public void loadFileFromDisc(String fileName){

        System.out.println("load file from disc ..");
    }

    @Override
    public void display(){
        System.out.println("display file ..."+fileName);
    }
}
