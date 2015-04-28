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

            float[][] A = new float[R][];
            float[][] B = new float[R][];
            float[][] C = new float[R][];

            Random rand = new Random();
            for (int row = 0; row < R; row++) {
                A[row] = new float[R];
                B[row] = new float[R];
                C[row] = new float[R];
                float[] a = A[row];
                float[] b = B[row];
                for (int col = 0; col < R; col++) {
                    a[col] = rand.Next(0, 100);
                    a[col] = rand.Next(0, 100);
                }
            }

            Parallel.For(0, R, (int row) => {
                float[] a = A[row];
                for (int col = 0; col < R; col++) {
                    float[] b = B[col];
                    float sum = 0;
                    for (int i = 0; i < R; i++) {
                        sum += a[i] * b[i];
                    }
                    C[row][col] = sum;
                }
            });

            if (args.Length > 1) {
                StreamWriter outa = new StreamWriter("a.txt");
                StreamWriter outb = new StreamWriter("b.txt");
                StreamWriter outc = new StreamWriter("c.txt");

                for (int row = 0; row < R; row++) {
                    for (int col = 0; col < R; col++) {
                        outa.Write(A[row][col]);
                        outa.Write(" ");
                        outb.Write(B[row][col]);
                        outb.Write(" ");
                        outc.Write(C[row][col]);
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
