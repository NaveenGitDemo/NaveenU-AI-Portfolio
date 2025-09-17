package com.chatmodel.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class WebChatController {

    @Autowired
    private HuggingFaceClientService huggingFaceClientService;

    @GetMapping("/")
    public String showChatPage() {
        return "chat";
    }

    @PostMapping("/webchat")
    public String handleUserMessage(@RequestParam("userMessage") String userMessage, Model model) {
        String botReply = huggingFaceClientService.getChatCompletionResponse(userMessage)
                .map(message -> message.getContent())
                .block(); // Blocking for simplicity in web context

        model.addAttribute("userMessage", userMessage);
        model.addAttribute("botReply", botReply);

        return "chat";
    }

}
