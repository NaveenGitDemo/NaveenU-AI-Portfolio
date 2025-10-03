package com.designe.patteren;

public class User {

    private int id;
    private String name;
    private String password;
    private String address;

    private User(Builder builder){
      this.id = builder.id;
      this.name = builder.name;
      this.password = builder.password;
      this.address = builder.address;
    }
    static public class Builder{

        private int id;
        private String name;
        private String password;
        private String address;

        public Builder id(int id){
           this.id = id;
           return this;
        }
        public Builder name(String name){
            this.name = name;
            return this;
        }
        public Builder password(String password){
            this.password = password;
            return this;
        }
        public Builder address(String address){
            this.address = address;
            return this;
        }

        public User build(){
            return new User(this);
        }
    }
    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", password='" + password + '\'' +
                ", address='" + address + '\'' +
                '}';
    }
}
