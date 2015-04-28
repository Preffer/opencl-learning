using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Matrix {
    class Program {
        static void Main(string[] args) {
            if (args.Length == 0) {
                Console.WriteLine("Usage: ./matrix (R)");
                Environment.Exit(2);
            }

            int R = int.Parse(args[0]);

            float[,] A = new float[R, R];
            float[,] B = new float[R, R];
            float[,] C = new float[R, R];

            Random rand = new Random();
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < R; j++) {
                    A[i, j] = rand.Next(0, 100);
                    B[i, j] = rand.Next(0, 100);
                }
            }

            for (int row = 0; row < R; row++) {
                for (int col = 0; col < R; col++) {
                    float sum = 0;
                    for (int i = 0; i < R; i++) {
                        sum += A[row, i] * B[i, col];
                    }
                    C[row, col] = sum;
                }
            }

            if (args.Length > 1) {
                StreamWriter outa = new StreamWriter("a.txt");
                StreamWriter outb = new StreamWriter("b.txt");
                StreamWriter outc = new StreamWriter("c.txt");

                for (int row = 0; row < R; row++) {
                    for (int col = 0; col < R; col++) {
                        outa.Write(A[row, col]);
                        outa.Write(" ");
                        outb.Write(B[row, col]);
                        outb.Write(" ");
                        outc.Write(C[row, col]);
                        outc.Write(" ");
                    }
                    outa.Write(Environment.NewLine);
                    outb.Write(Environment.NewLine);
                    outc.Write(Environment.NewLine);
                }
            }
        }
    }
}
