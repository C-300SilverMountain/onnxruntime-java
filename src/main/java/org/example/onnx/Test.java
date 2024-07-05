package org.example.onnx;

public class Test {
    public static void main(String[] args) {

        System.out.println(sigmoid(-10023.625D));
    }

    public static double sigmoid(double x) {
        x = Math.max(-500,x);
        x = Math.min(500,x);
        return 1 / (1 + Math.exp(-x));
    }
}
