package com.chatmodel.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = {"com.chatmodel.controller"})
public class ChatmodelApplication {

	public static void main(String[] args) {

		SpringApplication.run(ChatmodelApplication.class, args);
		System.out.println("Hello hugging face chat model : ");
	}

}
