package com.chatmodel.controller;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Map;


@RestController
@RequestMapping("/chat")
public class ChatController {

    private final HuggingFaceClientService huggingFaceClientService;


    public ChatController(HuggingFaceClientService huggingFaceClientService) {
        this.huggingFaceClientService = huggingFaceClientService;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<String> chat(@RequestBody String userInput) {
        return huggingFaceClientService.getChatResponse(userInput);
    }

    // new method for question answer type
    @PostMapping(path = "/generate", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Message> generateResp(@RequestBody String prompt) {
       Mono<Message> response = huggingFaceClientService.getChatCompletionResponse(prompt);
        return huggingFaceClientService.getChatCompletionResponse(prompt);
    }

}
