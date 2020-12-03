package edu.cuanschutz.ccp.mn_paper_baseline;

/*-
 * #%L
 * Colorado Computational Pharmacology's CRAFT Shared
 * 						Task Baseline Utility
 * 						project
 * %%
 * Copyright (C) 2020 Regents of the University of Colorado
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the Regents of the University of Colorado nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.tools.ant.util.StringUtils;
import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.cleartk.token.type.Sentence;

import com.google.common.annotations.VisibleForTesting;

import edu.ucdenver.ccp.common.collections.CollectionsUtil;
import edu.ucdenver.ccp.common.file.CharacterEncoding;
import edu.ucdenver.ccp.common.file.FileArchiveUtil;
import edu.ucdenver.ccp.common.file.FileWriterUtil;
import edu.ucdenver.ccp.common.file.reader.StreamLineIterator;
import edu.ucdenver.ccp.nlp.core.uima.annotation.CCPTextAnnotation;
import edu.ucdenver.ccp.nlp.pipelines.conceptmapper.ConceptMapperDictionaryFileFactory.DictionaryNamespace;
import edu.ucdenver.ccp.nlp.pipelines.conceptmapper.ConceptMapperPipelineFactory;
import edu.ucdenver.ccp.nlp.pipelines.conceptmapper.EntityFinder;
import edu.ucdenver.ccp.nlp.uima.util.TypeSystemUtil;
import edu.ucdenver.ccp.nlp.wrapper.conceptmapper.dictionary.obo.OboToDictionary.IncludeExt;

public class BaselineFileGenerator {

	private static final TypeSystemDescription TYPE_SYSTEM_DESCRIPTION = createConceptMapperTypeSystem();

	public enum Input {
		CORE, EXT
	}

	public enum Ontology {
		CHEBI("CHEBI.obo.zip", "CHEBI+extensions.obo.zip", DictionaryNamespace.CHEBI),
		CL("CL.obo.zip", "CL+extensions.obo.zip", DictionaryNamespace.CL),
		GO_BP("GO.obo.zip", "GO+GO_BP_extensions.obo.zip", DictionaryNamespace.GO_BP),
		GO_CC("GO.obo.zip", "GO+GO_CC_extensions.obo.zip", DictionaryNamespace.GO_CC),
		GO_MF("GO_MF_stub.obo.zip", "GO_MF_stub+GO_MF_extensions.obo.zip", DictionaryNamespace.GO_MF),
		MOP("MOP.obo.zip", "MOP+extensions.obo.zip", DictionaryNamespace.OBO),
		NCBITaxon("NCBITaxon.obo.zip", "NCBITaxon+extensions.obo.zip", DictionaryNamespace.NCBI_TAXON),
		PR("PR.obo.zip", "PR+extensions.obo.zip", DictionaryNamespace.PR),
		SO("SO.obo.zip", "SO+extensions.obo.zip", DictionaryNamespace.SO),
		UBERON("UBERON.obo.zip", "UBERON+extensions.obo.zip", DictionaryNamespace.OBO);

		private final String coreOntologyFilename;
		private final String extOntologyFilename;
		private final DictionaryNamespace dictNs;

		private Ontology(String coreOntologyFilename, String extOntologyFilename, DictionaryNamespace ns) {
			this.coreOntologyFilename = coreOntologyFilename;
			this.extOntologyFilename = extOntologyFilename;
			this.dictNs = ns;
		}

		private String getCraftConceptAnnotationDirectoryName(Input input) {
			return (input == Input.EXT) ? name() + "+extensions" : name();
		}

		/**
		 * @param craftBaseDirectory
		 * @param input
		 * @return a reference to the zipped ontology file from the CRAFT distribution
		 */
		public File getPackagedOntologyFile(File craftBaseDirectory, Input input) {
			String packagedFilename = (input == Input.CORE) ? this.coreOntologyFilename : this.extOntologyFilename;
			return new File(craftBaseDirectory, String.format("concept-annotation/%s/%s/%s", name(),
					getCraftConceptAnnotationDirectoryName(input), packagedFilename));
		}

		/**
		 * @param craftBaseDirectory
		 * @param input
		 * @param outputDirectory
		 * @return a reference to the unzipped ontology file, now in the specified
		 *         output directory
		 * @throws IOException
		 */
		public File getOntologyFile(File craftBaseDirectory, Input input, File outputDirectory) throws IOException {
			File packagedFile = getPackagedOntologyFile(craftBaseDirectory, input);
			String targetFilename = StringUtils.removeSuffix(packagedFile.getName(), ".zip");
			return FileArchiveUtil.unzip(packagedFile, outputDirectory, targetFilename);
		}

		public DictionaryNamespace getConceptMapperDictionaryNamespace() {
			return this.dictNs;
		}

	}

	private static final String SENTENCE_DETECTOR_TYPE_SYSTEM_STR = "org.cleartk.token.type.Sentence"; // "edu.ucdenver.ccp.nlp.ext.uima.annotators.sentencedetectors.TypeSystem";

	private static TypeSystemDescription createConceptMapperTypeSystem() {
		Collection<String> typeSystemStrs = new ArrayList<String>();
		typeSystemStrs.add(TypeSystemUtil.CCP_TYPE_SYSTEM);
		typeSystemStrs.add(SENTENCE_DETECTOR_TYPE_SYSTEM_STR);
		typeSystemStrs.addAll(ConceptMapperPipelineFactory.CONCEPTMAPPER_TYPE_SYSTEM_STRS);
		TypeSystemDescription tsd = TypeSystemDescriptionFactory
				.createTypeSystemDescription(typeSystemStrs.toArray(new String[typeSystemStrs.size()]));
		return tsd;
	}

	@VisibleForTesting
	protected static AnalysisEngine createConceptMapperEngine(Ontology ont, Input input, File dictionaryDirectory,
			File craftBaseDirectory) throws UIMAException, IOException {

		IncludeExt includeExt = IncludeExt.NO;
		if (input == Input.EXT) {
			includeExt = IncludeExt.YES;
		}
		
		/*
		 * The initialization code below builds a ConceptMapper dictionary for each
		 * ontology unless one already exists. Because UBERON and MOP annotations were
		 * added to CRAFT after the EntityFinder code was originally built, we use the
		 * generic "OBO" DictionaryNamespace when processing those two ontologies. The
		 * dictionary file created when using the OBO namespace is named cmDict-OBO.xml.
		 * Because the code can't easily tell if this dictionary contains MOP or UBERON
		 * data, we delete any existing cmDict-OBO.xml file each time and force the
		 * dictionary to be recreated.
		 */

		File oboDictFile = new File(dictionaryDirectory, "cmDict-OBO.xml");
		if (oboDictFile.exists()) {
			oboDictFile.delete();
		}

		List<AnalysisEngineDescription> conceptMapperAeDescriptions = EntityFinder
				.initConceptMapperAggregateDescriptions(TYPE_SYSTEM_DESCRIPTION,
						ont.getConceptMapperDictionaryNamespace().name(),
						ont.getOntologyFile(craftBaseDirectory, input, dictionaryDirectory), dictionaryDirectory,
						false, includeExt);

		AnalysisEngine conceptMapperEngine = AnalysisEngineFactory.createEngine(AnalysisEngineFactory
				.createEngineDescription(conceptMapperAeDescriptions.toArray(new AnalysisEngineDescription[0])));

		return conceptMapperEngine;
	}

	@VisibleForTesting
	protected static File getInputFile(File dataDirectory, Input input, Ontology ont) {
		String filename = String.format("gs_%s_%scombo_src_file.txt", ont.name(), (input == Input.EXT) ? "EXT_" : "");
		return new File(dataDirectory, String.format("input/%s/%s", input.name().toLowerCase(), filename));
	}

	@VisibleForTesting
	protected static File getOutputFile(File dataDirectory, Input input, Ontology ont) {
		String filename = String.format("gs_%s_%scombo_tgt_concept_ids.txt", ont.name(),
				(input == Input.EXT) ? "EXT_" : "");
		return new File(dataDirectory, String.format("output/%s/%s", input.name().toLowerCase(), filename));
	}

	/**
	 * Process the input file with ConceptMapper and generate the corresponding
	 * output file containing the identifiers corresponding to the input text.
	 * 
	 * @param conceptMapper
	 * @param inputFile
	 * @param outputFile
	 * @throws IOException
	 * @throws UIMAException
	 * @throws AnalysisEngineProcessException
	 * @throws FileNotFoundException
	 */
	private static void process(AnalysisEngine conceptMapper, File inputFile, File outputFile)
			throws IOException, UIMAException, AnalysisEngineProcessException, FileNotFoundException {
		try (BufferedWriter writer = FileWriterUtil.initBufferedWriter(outputFile)) {
			for (StreamLineIterator lineIter = new StreamLineIterator(inputFile, CharacterEncoding.UTF_8); lineIter
					.hasNext();) {
				String line = lineIter.next().getText();
				JCas jcas = processLine(conceptMapper, line);
				String outStr = extractMatchedIdentifiers(jcas);
				writer.write(outStr + "\n");
			}
		}
	}

	/**
	 * @param jcas
	 * @return a pipe-delimited string of matched ontology identifiers
	 */
	private static String extractMatchedIdentifiers(JCas jcas) {
		Set<String> ids = new HashSet<String>();
		for (Iterator<CCPTextAnnotation> annotIter = JCasUtil.iterator(jcas, CCPTextAnnotation.class); annotIter
				.hasNext();) {
			CCPTextAnnotation annotation = annotIter.next();
			ids.add(processId(annotation.getClassMention().getMentionName()));
		}
		List<String> sortedIds = new ArrayList<String>(ids);
		Collections.sort(sortedIds);
		String outStr = CollectionsUtil.createDelimitedString(sortedIds, "|");
		return outStr;
	}

	/**
	 * processes the input line with the ConceptMapper engine, adding annotations to
	 * the JCas for all matches
	 * 
	 * @param conceptMapper
	 * @param line
	 * @return
	 * @throws UIMAException
	 * @throws AnalysisEngineProcessException
	 */
	@VisibleForTesting
	protected static JCas processLine(AnalysisEngine conceptMapper, String line)
			throws UIMAException, AnalysisEngineProcessException {
		JCas jcas = JCasFactory.createJCas(TYPE_SYSTEM_DESCRIPTION);
		/*
		 * Note that a match will not be found unless there is a trailing space. We add
		 * a leading space just in case it is needed.
		 */
		line = " " + line + " ";
		jcas.setDocumentText(line);
		/*
		 * add a sentence annotation -- this is required by the ConceptMapper aggregate
		 * engine
		 */
		Sentence sentenceAnnot = new Sentence(jcas, 0, line.length());
		sentenceAnnot.addToIndexes();
		conceptMapper.process(jcas);
		return jcas;
	}

	/**
	 * @param mentionName
	 * @return id with OBO PURL removed
	 */
	@VisibleForTesting
	protected static String processId(String id) {
		if (id.startsWith("http://purl.obolibrary.org/obo/")) {
			id = StringUtils.removePrefix(id, "http://purl.obolibrary.org/obo/");
		}
		
		if (id.contains("NCBITaxon_EXT_")) {
			id = id.replace("NCBITaxon_EXT_", "NCBITaxon_EXT:");
		} else if (id.contains("EXT_")) {
			id = id.replace("EXT_", "EXT:");
		} else if (id.contains("EXT#")) {
			id = id.replace("EXT#_", "EXT:");
		} else if (id.contains("_")) {
			id = id.replace("_", ":");
		}
		return id;
	}

	/**
	 * mvn exec:java
	 * -Dexec.mainClass=edu.cuanschutz.ccp.mn_paper_baseline.BaselineFileGenerator
	 * -Dexec.args="/home/baseline/CRAFT-4.0.1 /home/baseline/dictionaries
	 * /home/baseline/data"
	 * 
	 * @param args args[0] = craft base directory path <br>
	 *             s args[1] = dictionary directory path
	 */
	public static void main(String[] args) {

		File craftBaseDirectory = new File(args[0]);
		File dictionaryDirectoryBase = new File(args[1]);
		File dataDirectory = new File(args[2]);

		try {
			// this block used for testing
//			{
//				File craftBaseDirectory = new File("/Users/bill/projects/craft-shared-task/craft.git");
//				File dictionaryDirectory = new File(
//						"/Users/bill/projects/one-offs/for-mayla-negacy-concept-recognition-paper/dictionaries");
//				File dataDirectory = new File(
//						"/Users/bill/projects/one-offs/for-mayla-negacy-concept-recognition-paper/data");
//				Ontology ont = Ontology.GO_MF;
//				Input input = Input.EXT;
//
//				// create ConceptMapper instance
//				AnalysisEngine conceptMapper = createConceptMapperEngine(ont, input, dictionaryDirectory,
//						craftBaseDirectory);
//
//				// get input file
//				File inputFile = getInputFile(dataDirectory, input, ont);
//				File outputFile = getOutputFile(dataDirectory, input, ont);
//
//				// process input file and create output file
//				process(conceptMapper, inputFile, outputFile);
//			}

			for (Input input : Input.values()) {
				File dictionaryDirectory = new File(dictionaryDirectoryBase, input.name());
				dictionaryDirectory.mkdir();
				for (Ontology ont : Ontology.values()) {

					System.out.println("PROCESSING " + input.name() + " -- " + ont.name());

					// create ConceptMapper instance
					AnalysisEngine conceptMapper = createConceptMapperEngine(ont, input, dictionaryDirectory,
							craftBaseDirectory);

					// get input file
					File inputFile = getInputFile(dataDirectory, input, ont);
					File outputFile = getOutputFile(dataDirectory, input, ont);

					// process input file and create output file
					process(conceptMapper, inputFile, outputFile);
				}
			}
		} catch (IOException | UIMAException e) {
			e.printStackTrace();
			System.exit(-1);
		}

	}

}
