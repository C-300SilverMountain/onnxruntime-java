package org.example.controller;

import com.alibaba.fastjson.JSONObject;
import lombok.extern.slf4j.Slf4j;
import org.example.onnx.RunUbertaOnOnnx;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * @Author dan.shuai
 * @Classname dd
 * @Description 说明
 * @Create 2023-12-06 13:35
 */
@Slf4j
@RestController
@RequestMapping("/ner")
public class HelpController {

    @RequestMapping(value = "/predict")
    public Object proSeg(@RequestParam(value = "query", defaultValue = "") String query) {
        try {
            List<JSONObject> predict = RunUbertaOnOnnx.predict(query);
            return predict;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }


}
