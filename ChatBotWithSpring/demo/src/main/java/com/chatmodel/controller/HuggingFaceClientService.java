package com.chatmodel.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

@Service
public class HuggingFaceClientService {

    private final WebClient webClient;

    public HuggingFaceClientService(@Value("${huggingface.api-url}") String apiUrl,
                             @Value("${huggingface.api-key}") String apiKey) {
        this.webClient = WebClient.builder()
                .baseUrl(apiUrl)
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                .build();
    }

    public Mono<String> getChatResponse(String prompt) {
        Map<String, String> body = Map.of("inputs", prompt);

        return webClient.post()
                .bodyValue(body)
                .retrieve()
                .bodyToMono(String.class);
    }
    // create one more method for the question and answer type prompt
    public Mono<Message> getChatCompletionResponse(String userMessage) {
        Map<String, Object> requestBody = Map.of(
                "model", "moonshotai/Kimi-K2-Instruct:together",
                "messages", List.of(
                        Map.of("role", "user", "content", userMessage)
                ),
                "stream", false
        );

        System.out.println("request body :"+requestBody);
        return webClient.post()
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(ChatResponse.class)
                .map(resp -> resp.getChoices().get(0).getMessage());
    }
}
