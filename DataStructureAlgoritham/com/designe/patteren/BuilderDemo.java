package com.designe.patteren;

public class BuilderDemo {

    public static void main(String[] arg){

        User.Builder builder = new User.Builder();
       User user = builder.id(1).name("Krishn").address("Gokul").password("Rukmani").build();

       System.out.println(user);
    }
}
