import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

import opennlp.tools.stemmer.PorterStemmer;

public class Problem2A {

    public static class DocumentFrequencyMapper extends Mapper<LongWritable, Text, Text, Text> {

        private Text term = new Text();
        private Text docIdText = new Text();
        private Set<String> stopwords = new HashSet<>();
        private PorterStemmer stemmer;
        private String docId;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            stemmer = new PorterStemmer();

            // Load stopwords
            URI[] cacheFiles = Job.getInstance(conf).getCacheFiles();
            if (cacheFiles != null && cacheFiles.length > 0) {
                for (URI cacheURI : cacheFiles) {
                    Path path = new Path(cacheURI.getPath());
                    if (path.getName().equals("stopwords.txt")) {
                        try (BufferedReader reader = new BufferedReader(new FileReader(path.getName()))) {
                            String word;
                            while ((word = reader.readLine()) != null) {
                                stopwords.add(word.trim().toLowerCase());
                            }
                        } catch (IOException e) {
                            System.err.println("Error reading stopwords file: " + e.getMessage());
                        }
                    }
                }
            }

            // Get Document ID from FileSplit
            FileSplit split = (FileSplit) context.getInputSplit();
            docId = split.getPath().getName();
            docIdText.set(docId);
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z]+");
            
            Set<String> uniqueTermsInLine = new HashSet<>();

            for (String token : tokens) {
                if (token.length() > 0 && !stopwords.contains(token)) {
                    String stemmed = stemmer.stem(token);
                    if (stemmed.length() > 0) {
                        uniqueTermsInLine.add(stemmed);
                    }
                }
            }

            for (String uniqueTerm : uniqueTermsInLine) {
                term.set(uniqueTerm);
                context.write(term, docIdText);
            }
        }
    }

    public static class DocumentFrequencyReducer extends Reducer<Text, Text, Text, IntWritable> {

        private IntWritable dfWritable = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Set<String> uniqueDocs = new HashSet<>();
            for (Text val : values) {
                uniqueDocs.add(val.toString());
            }
            dfWritable.set(uniqueDocs.size());
            context.write(key, dfWritable);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: Problem2A <input path> <output path> [-stopwords <stopwords file>]");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        // Use a tab separator for the output
        conf.set("mapreduce.output.textoutputformat.separator", "\t");
        
        Job job = Job.getInstance(conf, "Document Frequency");
        job.setJarByClass(Problem2A.class);

        job.setMapperClass(DocumentFrequencyMapper.class);
        job.setReducerClass(DocumentFrequencyReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class); // Mapper output value

        int fileArgIndex = 0;
        for (int i = 0; i < args.length; ++i) {
            if ("-stopwords".equals(args[i])) {
                job.addCacheFile(new Path(args[++i]).toUri());
            } else if (fileArgIndex == 0) {
                FileInputFormat.addInputPath(job, new Path(args[i]));
                fileArgIndex++;
            } else if (fileArgIndex == 1) {
                FileOutputFormat.setOutputPath(job, new Path(args[i]));
                fileArgIndex++;
            }
        }

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
