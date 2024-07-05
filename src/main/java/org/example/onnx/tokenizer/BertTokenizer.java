package org.example.onnx.tokenizer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import cn.hutool.core.collection.ListUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import lombok.extern.log4j.Log4j2;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Bert Tokenizer
 *
 * @author jadepeng
 */
@Log4j2
public class BertTokenizer implements Tokenizer {

    private String vocabFile = "vocab.txt";
    private Map<String, Integer> tokenIdMap;
    private Map<Integer, String> idTokenMap;
    private final boolean doLowerCase = true;
    private final boolean doBasicTokenize = true;
    private final List<String> neverSplit = new ArrayList<String>();
    private final String unkToken = "[UNK]";
    private final String sepToken = "[SEP]";
    private final String padToken = "[PAD]";
    private final String clsToken = "[CLS]";
    private final String maskToken = "[MASK]";
    private final boolean tokenizeChineseChars = true;
    private BasicTokenizer basicTokenizer;
    private WordpieceTokenizer wordpieceTokenizer;

    private static final int MAX_LEN = 2048;

    public BertTokenizer(String vocabFile) {
        this.vocabFile = vocabFile;
        init();
    }

    public BertTokenizer() {
        init();
    }

    private void init() {
        try {
            this.tokenIdMap = loadVocab(vocabFile);
        } catch (IOException e) {
            log.error("Unable to load vocab due to: ", e);
        }
        this.idTokenMap = new HashMap<>(this.tokenIdMap.size());
        for (String key : tokenIdMap.keySet()) {
            this.idTokenMap.put(tokenIdMap.get(key), key);
        }

        if (doBasicTokenize) {
            this.basicTokenizer = new BasicTokenizer(doLowerCase, neverSplit, tokenizeChineseChars);
        }
        this.wordpieceTokenizer = new WordpieceTokenizer(tokenIdMap, unkToken);
    }

    private Map<String, Integer> loadVocab(String vocabFileName) throws IOException {
        return TokenizerUtils.generateTokenIdMap(new FileInputStream(vocabFileName));
    }

    /**
     * Tokenizes a piece of text into its word pieces.
     * <p>
     * This uses a greedy longest-match-first algorithm to perform tokenization
     * using the given vocabulary.
     * <p>
     * For example: input = "unaffable" output = ["un", "##aff", "##able"]
     * <p>
     * Args: text: A single token or whitespace separated tokens. This should have
     * already been passed through `BasicTokenizer`.
     * <p>
     * Returns: A list of wordpiece tokens.
     */
    @Override
    public List<String> tokenize(String text) {
        List<String> splitTokens = new ArrayList<>();
        if (doBasicTokenize) {
            for (String token : basicTokenizer.tokenize(text)) {
                splitTokens.addAll(wordpieceTokenizer.tokenize(token));
            }
        } else {
            splitTokens = wordpieceTokenizer.tokenize(text);
        }
        return splitTokens;
    }

    public String convertTokensToString(List<String> tokens) {
        // Converts a sequence of tokens (string) in a single string.
        return tokens.stream().map(s -> s.replace("##", "")).collect(Collectors.joining(" "));
    }

    public long[][] convertTokensToIds(List<String> tokens) {
        long[][] result = new long[1][];
        int i = 1;
        for (String s : tokens) {
            result[0][i++] = tokenIdMap.get(s);
        }
        result[0][i++] = tokenIdMap.get(sepToken);
        return result;
    }

    static long[] paddingZero(long[] array1, int paddingSize) {
        long[] result = new long[array1.length + paddingSize];
        System.arraycopy(array1, 0, result, 0, array1.length);
        for (int i = array1.length; i < result.length; i++) {
            result[i] = 0;
        }
        return result;
    }

    public Map<String, OnnxTensor> tokenizeOnnxTensor(List<String> texts)
            throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        long[][] textTokensIds = new long[texts.size()][];
        long[][] masks = new long[texts.size()][];
        int rowIndex = 0;
        int maxColumn = 32;
        for (String text : texts) {

            List<String> tokens = tokenize(text);
            long[] tokenIds = new long[tokens.size() + 2];
            int index = 0;
            tokenIds[index++] = tokenIdMap.get(clsToken);
            for (String s : tokens) {
                tokenIds[index++] = tokenIdMap.get(s);
            }
            tokenIds[index++] = tokenIdMap.get(sepToken);
            textTokensIds[rowIndex] = tokenIds;
            masks[rowIndex++] = buildTokenTypeArray(index);
            maxColumn = Math.max(maxColumn, index);
        }

        // 长度不足maxColumn，填充0，必须保证长度达maxColumn
        for (int row = 0; row < texts.size(); row++) {
            if (textTokensIds[row].length < maxColumn) {
                // padding 0
                textTokensIds[row] = paddingZero(textTokensIds[row], maxColumn - textTokensIds[row].length);
                masks[row] = paddingZero(masks[row], maxColumn - masks[row].length);
            }
        }

        OnnxTensor ids = OnnxTensor.createTensor(env, new long[][]{textTokensIds[0]});
        OnnxTensor tokenTypeIds = OnnxTensor.createTensor(env, new long[][]{masks[0]});
        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("ids", ids);
        inputMap.put("mask", tokenTypeIds);


        return inputMap;
    }

    public Map<String, OnnxTensor> tokenizeOnnxTensorForRoberta(List<String> texts)
            throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        long[][] textTokensIds = new long[texts.size()][];
        long[][] masks = new long[texts.size()][];
        long[][] types = new long[texts.size()][];
        int rowIndex = 0;
        int maxColumn = 300;
        for (String text : texts) {

            List<String> tokens = tokenize(text);
            long[] tokenIds = new long[tokens.size() + 2];
            int index = 0;
            tokenIds[index++] = tokenIdMap.get(clsToken);
            for (String s : tokens) {
                tokenIds[index++] = tokenIdMap.get(s);
            }
            tokenIds[index++] = tokenIdMap.get(sepToken);
            textTokensIds[rowIndex] = tokenIds;
            types[rowIndex] = buildTypeArray(maxColumn);

            masks[rowIndex++] = buildTokenTypeArray(index);
            maxColumn = Math.max(maxColumn, index);
        }

        // 长度不足maxColumn，填充0，必须保证长度达maxColumn
        for (int row = 0; row < texts.size(); row++) {
            if (textTokensIds[row].length < maxColumn) {
                // padding 0
                textTokensIds[row] = paddingZero(textTokensIds[row], maxColumn - textTokensIds[row].length);
                masks[row] = paddingZero(masks[row], maxColumn - masks[row].length);
            }
        }

        OnnxTensor ids = OnnxTensor.createTensor(env, new long[][]{textTokensIds[0]});
        OnnxTensor attentionMask = OnnxTensor.createTensor(env, new long[][]{masks[0]});
        OnnxTensor tokenType = OnnxTensor.createTensor(env, new long[][]{types[0]});
        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("input_ids", ids);
        inputMap.put("attention_mask", attentionMask);
        inputMap.put("token_type_ids", tokenType);

        return inputMap;
    }

    private double[][] zeros(int maxLength){
        double[][] result = new double[50][];
        for (int i = 0; i < maxLength; i++) {
            double[] rs = buildMashArray(maxLength);
            result[i] = rs;
        }
        return result;
    }

    public void decode(double[][][][]spanLogits, long[][]pos, List<JSONObject> inpBatchData){

        for (int i = 0; i < inpBatchData.size(); i++) {
            JSONObject item = inpBatchData.get(i);

            String textb = item.getString("text");

            List<String> tokenize = tokenize(textb);
            long[][]offsetMapping = new long[tokenize.size()][];
            for (int j = 0; j < tokenize.size(); j++) {
                long[] t = {j};
                offsetMapping[j] = t;
            }


            JSONArray choices = item.getJSONArray("choices");


            for (int c = 0; c < choices.size(); c++) {

                String texta = item.getString("task_type") + "[SEP]"
                        + item.getString("subtask_type") + "[SEP]"
                        + choices.getJSONObject(c).getString("entity_type");
                Long[] encode = encode(texta);
                int text_start_id = encode.length;
                List<double[]>entity_idx_type_list = new ArrayList<>();
                for(long[] entitySpan: pos){
                    if (entitySpan[0] == i && entitySpan[1] == c){
                        int l2 = (int) entitySpan[2];
                        int l3 = (int) entitySpan[3];
                        double prob = spanLogits[0][c][l2][l3];

                        entity_idx_type_list.add(new double[]{(double)entitySpan[2], (double)entitySpan[3], prob});
                    }
                }

                JSONArray entity_list = new JSONArray();
                List<String>dupMap = new ArrayList<>();
                for(double[]entity_idx:entity_idx_type_list ){
                    String entity = extract_entity(textb, entity_idx, text_start_id, offsetMapping);

                    if (!dupMap.contains(entity)){
                        dupMap.add(entity);

                        JSONObject entityObj = new JSONObject();
                        entityObj.put("entity_name",entity);
                        entityObj.put("score",entity_idx[2]);
                        entity_list.add(entityObj);
                    }

                }

                System.out.println(i+": "+c+": "+JSON.toJSONString(entity_list));
//                batch_data[i]['choices'][c]['entity_list'] = entity_list
                JSONObject target = choices.getJSONObject(c);
                target.put("entity_list",entity_list);

            }
        }

        System.out.println("");
    }

    private String extract_entity(String text, double[] entityIdx, int textStartId, long[][] offsetMapping) {

        long[] start_split = (entityIdx[0] - textStartId) < offsetMapping.length && entityIdx[0] - textStartId >= 0
                ?offsetMapping[(int) entityIdx[0] - textStartId]:new long[]{};

        long[]end_split = offsetMapping.length>(entityIdx[1] - textStartId) && (entityIdx[1] - textStartId)>=0?
                offsetMapping[(int)entityIdx[1] - textStartId]:new long[]{};

        String entity = "";
        if (start_split.length>0 && end_split.length>0){
            entity = text.substring((int)start_split[0],(int)end_split[end_split.length-1]+1);
        }
        return entity;
    }

    private Long[] encode(String text){
        String[] split = text.split("\\[SEP\\]");
        List<Integer>tokenIds = new ArrayList<>();
        tokenIds.add(tokenIdMap.get(clsToken));
        for (int i = 0; i < split.length; i++) {
            List<String> tokens = tokenize(split[i]);
            for (String s : tokens) {
                tokenIds.add(tokenIdMap.get(s));
            }
            tokenIds.add(tokenIdMap.get(sepToken));
        }
//        tokenIds.add(tokenIdMap.get(sepToken));
        Long[] t = new Long[tokenIds.size()];
        for (int i = 0; i < tokenIds.size(); i++) {
            t[i] = Long.valueOf(tokenIds.get(i));
        }
        return t;
    }



    public Map<String, OnnxTensor> encode(List<JSONObject> jsonObjects )
            throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        long[][][] allInputIds = new long[1][][];
        long[][][] allAttentionMask = new long[1][][];
        long[][][] allTokenTypeIds = new long[1][][];
        double[][][][]allSpanLabelMask = new double[1][][][];
        int rowIndex = 0;
        int maxLength = 50;
        for (JSONObject reqBody : jsonObjects) {

            long[][] allInputIds0 = new long[6][];
            long[][] allAttentionMask0 = new long[6][];
            long[][] allTokenTypeIds0 = new long[6][];
            double[][][]allSpanLabelMask0 = new double[6][][];

            String text = reqBody.getString("text");
            JSONArray choices = reqBody.getJSONArray("choices");
            for (int i = 0; i < choices.size(); i++) {
                JSONObject choose = choices.getJSONObject(i);
                String entityType = choose.getString("entity_type");
                String texta = "抽取任务-实体识别-"+entityType+"-"+text;
                List<String> tokens = tokenize(texta);
                tokens = tokens.stream().map(e-> "-".equals(e)?sepToken:e).collect(Collectors.toList());
                int index = 0;
                long[] inputIds = new long[maxLength];
                inputIds[index++] = tokenIdMap.get(clsToken);
                for (String s : tokens) {
                  inputIds[index++] = tokenIdMap.get(s);
                }
                inputIds[index] = tokenIdMap.get(sepToken);

                long[] tokenTypeIds = buildTypeArray(maxLength);
                for(int j=index;j>0;j--){
                    if (j<index && inputIds[j]==102){
                        break;
                    }
                    tokenTypeIds[j] = 1;
                }

                long[]attentionMask= buildTypeArray(maxLength);
                for (int j = 0; j < inputIds.length; j++) {
                    if (inputIds[j] == 0){
                        break;
                    }
                    attentionMask[j]=1;
                }
                Long[] encode = encode("抽取任务[SEP]实体识别[SEP]" + entityType);
                double[][]spanLabelMask =zeros(maxLength);
                for (int j = 0; j <spanLabelMask.length; j++) {
                    if (j<encode.length){
                        continue;
                    }
                    double []row = spanLabelMask[j];
                    for (int k = encode.length; k <row.length; k++) {
                        row[k] = 0;
                    }
                }

                allInputIds0[rowIndex] = inputIds;
                allTokenTypeIds0[rowIndex] = tokenTypeIds;
                allAttentionMask0[rowIndex] = attentionMask;
                allSpanLabelMask0[rowIndex] = spanLabelMask;
                rowIndex++;
            }

            allInputIds[0] = allInputIds0;
            allTokenTypeIds[0] = allTokenTypeIds0;
            allAttentionMask[0] = allAttentionMask0;
            allSpanLabelMask[0] = allSpanLabelMask0;
        }

        OnnxTensor ids = OnnxTensor.createTensor(env, allInputIds);
        OnnxTensor attentionMask = OnnxTensor.createTensor(env, allAttentionMask);
        OnnxTensor tokenType = OnnxTensor.createTensor(env, allTokenTypeIds);
        OnnxTensor spanLabelMask = OnnxTensor.createTensor(env, allSpanLabelMask);
        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("input_ids", ids);
        inputMap.put("attention_mask", attentionMask);
        inputMap.put("token_type_ids", tokenType);
        inputMap.put("span_labels_mask", spanLabelMask);

        return inputMap;
    }

    long[] buildTokenTypeArray(int size) {
        long[] mask = new long[size];
        for (int i = 0; i < size; i++) {
            mask[i] = 1;
        }
        return mask;
    }

    long[] buildTypeArray(int size) {
        long[] mask = new long[size];
        for (int i = 0; i < size; i++) {
            mask[i] = 0;
        }
        return mask;
    }

    double[] buildMashArray(int size) {
        double[] mask = new double[size];
        for (int i = 0; i < size; i++) {
            mask[i] = -10000.0D;
        }
        return mask;
    }

    public int vocabSize() {
        return tokenIdMap.size();
    }
}
