import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Problem1E_MapClassLevel_Pairs {

    // =========================
    // MAPPER — Map-Class Level Aggregation
    // Accumulates counts in HashMap, emits once in cleanup()
    // =========================
    public static class PairMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> topWords = new HashSet<>();
        private int window;
        private Map<String, Integer> pairCounts = new HashMap<>(); // ← class-level aggregation

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            window = conf.getInt("window", 1);

            URI[] files = context.getCacheFiles();
            if (files == null || files.length == 0) {
                throw new IOException("Top-50 cache file is missing.");
            }

            File localizedFile = new File(new Path(files[0].getPath()).getName());
            if (!localizedFile.exists()) {
                localizedFile = new File(files[0].getPath());
            }
            if (!localizedFile.exists()) {
                throw new IOException("Could not locate top-50 cache file: " + files[0]);
            }

            try (BufferedReader br = new BufferedReader(new FileReader(localizedFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length == 0 || parts[0].isEmpty()) continue;
                    topWords.add(parts[0].toLowerCase());
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^\\w']+");

            for (int i = 0; i < tokens.length; i++) {

                if (tokens[i].isEmpty() || !topWords.contains(tokens[i])) continue;

                for (int j = Math.max(0, i - window);
                     j <= Math.min(tokens.length - 1, i + window);
                     j++) {

                    if (i == j) continue;
                    if (tokens[j].isEmpty() || !topWords.contains(tokens[j])) continue;

                    String pair = "(" + tokens[i] + "," + tokens[j] + ")";
                    // ← Accumulate locally, don't emit yet
                    pairCounts.put(pair, pairCounts.getOrDefault(pair, 0) + 1);
                }
            }
        }

        @Override
        protected void cleanup(Context context)
                throws IOException, InterruptedException {
            // ← Emit all aggregated counts at end of mapper
            for (Map.Entry<String, Integer> entry : pairCounts.entrySet()) {
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
            }
        }
    }

    // =========================
    // REDUCER
    // =========================
    public static class PairReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable v : values) {
                sum += v.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    // =========================
    // MAIN
    // =========================
    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.err.println("Usage: Problem1E_MapClassLevel_Pairs <input> <output> <top50file> <window>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.setInt("window", Integer.parseInt(args[3]));

        long startTime = System.currentTimeMillis();

        Job job = Job.getInstance(conf, "Pairs Co-occurrence - Map Class Level");
        job.setJarByClass(Problem1E_MapClassLevel_Pairs.class); // ✅ correct class

        job.setMapperClass(PairMapper.class);
        job.setReducerClass(PairReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.addCacheFile(new Path(args[2]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean success = job.waitForCompletion(true);

        long endTime = System.currentTimeMillis();
        System.err.println("=== TOTAL RUNTIME (sec): " + (endTime - startTime) / 1000.0 + " ===");

        System.exit(success ? 0 : 1);
    }
}